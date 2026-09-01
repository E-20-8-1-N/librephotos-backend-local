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
if [[ "${DISABLE_CUDA_CACHE:-0}" = "1" ]]; then
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
fi

set -e

echo "LibrePhotos starting..."

# Matplotlib comes along with insightface, which the face recognition service
# imports. Left to itself it keeps its font cache under $HOME, and when the home
# directory is not writable it falls back to a fresh /tmp directory and re-parses
# every font on each start. Keep it on our own volume instead; this is the same
# path librephotos/settings/production.py derives.
mpl_data_root="${BASE_DATA:-/}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${mpl_data_root%/}/protected_media/matplotlib}"
mkdir -p "$MPLCONFIGDIR" || echo "Could not create $MPLCONFIGDIR - matplotlib will rebuild its font cache on every start"

# BASE_LOGS is what the settings module derives LOGS_ROOT and secret.key from.
# Export it so the tee targets below and Django's own log
# handlers cannot drift apart; pinning the default here leaves the container's
# behaviour unchanged for anyone who never sets it. LOG_LEVEL is read by the
# LOGGING dictConfig. The %/ handles a value written with the trailing slash
# the Python default uses, so "$logs_dir/x" does not become "//x".
export BASE_LOGS="${BASE_LOGS:-/logs}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
logs_dir="${BASE_LOGS%/}"

# This image never created the directory at all, so a plain `docker run` with no
# /logs volume died on the first secret.key write. Unlike the matplotlib cache
# above the directory is not optional, so name the path instead of continuing.
if ! mkdir -p "$logs_dir"; then
    echo "Could not create $logs_dir - set BASE_LOGS to a writable directory" >&2
    exit 1
fi

# Check if we should serve frontend
if echo "$SERVE_FRONTEND" | grep -qiE '^(true|1|yes|on)$'; then
    echo "Configuring for no-proxy deployment (serving frontend from Django)..."
    
    # Select the no-proxy settings module. It imports librephotos.settings.
    # production and overrides only what serving the frontend from Django
    # changes; this used to be a `cp` of a second, hand-maintained copy of
    # production.py over the real one, which drifted and broke /api/sitesettings.
    # manage.py and wsgi.py both use os.environ.setdefault, so exporting here
    # wins for every python invocation below, for gunicorn, and for the ML
    # services those spawn.
    export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-librephotos.settings.production_noproxy}"
    
    # Collect static files including frontend
    echo "Collecting static files..."
    python manage.py collectstatic --noinput

    echo "Frontend will be served from Django on port 8001"
else
    echo "Using standard proxy setup..."
fi

# Run migrations based on database backend
echo "Running migrations..."
DB_BACKEND=${DB_BACKEND:-sqlite}

run_migrations() {
    set +e
    python manage.py migrate 2>&1 | tee /logs/command_migrate.log
    migrate_status=${PIPESTATUS[0]}
    set -e

    if [ "$migrate_status" -ne 0 ]; then
        if grep -qi "Conflicting migrations detected" /logs/command_migrate.log; then
            echo "Conflicting migrations detected. Attempting auto-merge..."
            python manage.py makemigrations --merge --noinput 2>&1 | tee /logs/command_makemigrations_merge.log
            python manage.py migrate 2>&1 | tee /logs/command_migrate_retry.log
        else
            echo "Migration failed for another reason."
            exit "$migrate_status"
        fi
    fi
}

if [ "$DB_BACKEND" = "sqlite" ]; then
    echo "Using production-optimized SQLite database mode"
    # Ensure database directory exists
    mkdir -p /data/db
    # Run migrations for both default and cache databases
    run_migrations
    
elif [ "$DB_BACKEND" = "postgresql" ]; then
    echo "Using PostgreSQL database mode"
    
    python manage.py showmigrations | tee /logs/show_migrate.log
    # Run standard migrations
    run_migrations
    python manage.py showmigrations | tee /logs/show_migrate.log
else
    echo "Error: Unsupported DB_BACKEND: $DB_BACKEND"
    echo "Supported values: sqlite, postgresql"
    exit 1
fi

# Create cache directory
mkdir -p /root/.cache

# Check if we need to create a superuser
if [ ! -z "$ADMIN_USERNAME" ] && [ ! -z "$ADMIN_PASSWORD" ]; then
    echo "Creating/updating admin user..."
    python manage.py shell <<EOF
from api.models import User
from django.contrib.auth.hashers import make_password
import os

username = os.environ.get('ADMIN_USERNAME')
password = os.environ.get('ADMIN_PASSWORD')
email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')

try:
    user = User.objects.get(username=username)
    user.set_password(password)
    user.email = email
    user.save()
    print(f"Updated existing admin user: {username}")
except User.DoesNotExist:
    user = User.objects.create_superuser(username=username, password=password, email=email)
    print(f"Created new admin user: {username}")
EOF
fi

echo "Starting Django server..."

python manage.py start_service all
python manage.py start_cleaning_service
python manage.py start_job_cleanup_service
python manage.py clear_cache 
python manage.py build_similarity_index 2>&1 | tee "$logs_dir/command_build_similarity_index.log"

if [[ -n "$ADMIN_USERNAME" ]]; then
    python manage.py createadmin -u "$ADMIN_USERNAME" "$ADMIN_EMAIL" 2>&1 | tee /logs/command_createadmin.log
fi

echo "Running backend server..."

python manage.py qcluster 2>&1 | tee "$logs_dir/qcluster.log" &

GUNICORN_WORKERS=${WEB_CONCURRENCY:-1}

# Start the Django server
if [ "$DEBUG" = "1" ]; then
    python manage.py runserver 0.0.0.0:8001
else
    # Production server with gunicorn
    gunicorn --bind 0.0.0.0:8001 --worker-class=gevent --workers=$GUNICORN_WORKERS --timeout 3600 --max-requests 2000 --max-requests-jitter 50 --log-level=info librephotos.wsgi:application 2>&1 | tee /logs/gunicorn_django.log
fi 
