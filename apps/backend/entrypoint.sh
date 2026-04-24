#!/bin/bash
export PYTHONUNBUFFERED=TRUE
export PYTHONFAULTHANDLER=1

if [[ "$(uname -m)" == "aarch64"* ]]; then
    export OPENBLAS_CORETYPE=ARMV8
    echo "ARM architecture detected. OPENBLAS_CORETYPE set to ARMV8"
fi
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1

# --- Memory optimization ---
# Force glibc to return freed memory to the OS more aggressively
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
# Limit PyTorch/MKL threads to reduce per-thread memory overhead.
# With multiple subprocess services (clip, face, thumbnail, exif,
# image_similarity) + gunicorn + qcluster, every extra thread per
# process multiplies RSS and PID count. 1 is the safe default for
# CPU-only deployments; override per-service if needed.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
# ONNX Runtime (insightface) has its own thread pools.
export ORT_NUM_THREADS=${ORT_NUM_THREADS:-1}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-1}
# Disable PyTorch gradient tracking globally (inference only)
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

mkdir -p /logs
python manage.py showmigrations | tee /logs/show_migrate.log
python manage.py migrate | tee /logs/command_migrate.log
python manage.py showmigrations | tee /logs/show_migrate.log
python manage.py collectstatic --no-input
python manage.py start_service all
python manage.py start_cleaning_service
python manage.py clear_cache 
python manage.py build_similarity_index 2>&1 | tee /logs/command_build_similarity_index.log

if [[ -n "$ADMIN_USERNAME" ]]; then
    python manage.py createadmin -u "$ADMIN_USERNAME" "$ADMIN_EMAIL" 2>&1 | tee /logs/command_createadmin.log
fi

# Use pre-downloaded captioning model in HF_HOME if it exists
export HF_HOME="${HF_HOME:-/mnt/cetapod-suite/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

mkdir -p "$HF_HOME"

echo "Running backend server..."

python manage.py qcluster 2>&1 | tee /logs/qcluster.log &

GUNICORN_WORKERS=${WEB_CONCURRENCY:-1}

if [[ "$DEBUG" = 1 ]]; then
    echo "development backend starting"
    gunicorn --worker-class=gevent --workers=$GUNICORN_WORKERS --max-requests 50 --max-requests-jitter 10 --reload --bind 0.0.0.0:8001 --log-level=info librephotos.wsgi 2>&1 | tee /logs/gunicorn_django.log
else
    echo "production backend starting"
    gunicorn --worker-class=gevent --workers=$GUNICORN_WORKERS --max-requests 50 --max-requests-jitter 10 --bind 0.0.0.0:8001 --log-level=info librephotos.wsgi 2>&1 | tee /logs/gunicorn_django.log
fi