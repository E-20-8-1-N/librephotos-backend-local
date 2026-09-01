FROM cetapod/librephotos_backend_base:latest
# actual project

ARG DEBUG
ARG MLVALIDATION
ARG IMAGE_TAG
ENV IMAGE_TAG=${IMAGE_TAG}

# Install necessary packages
RUN apt-get update && apt-get install -y \
    nano \
    vim

# Install HEIF support image libraries
RUN apt-get install -y \
    libheif1 \
    libde265-0 \
    libde265-dev \
    x265 \
    libvips \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
# Build context is the monorepo root; source lives under apps/backend/.
COPY apps/backend/ /code/
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt
RUN if [ "$DEBUG" = 1 ] ; then \
    pip install --break-system-packages -r requirements.dev.txt; \
    fi
RUN if [ "$MLVALIDATION" = 1 ] ; then \
    apt-get update && apt-get install default-jre -y; \
    pip install --break-system-packages -r requirements.mlval.txt; \
    fi
EXPOSE 8001

COPY deploy/docker/backend/entrypoint.sh /entrypoint.sh
CMD ["/entrypoint.sh"]
