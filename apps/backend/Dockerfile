# FROM reallibrephotos/librephotos-base:dev
# actual project
FROM ethanhph/librephotos_local_base:latest

ARG DEBUG
ARG MLVALIDATION
ARG IMAGE_TAG
ENV IMAGE_TAG=${IMAGE_TAG}

WORKDIR /code
# Define GIT_USERNAME for PAT authentication
# ARG GIT_USERNAME=x-access-token
# RUN --mount=type=secret,id=github_pat \
#     git clone https://${GIT_USERNAME}:$(cat /run/secrets/github_pat)@github.com/Wiheim/librephotos_test.git .
RUN git clone --branch develop --single-branch https://github.com/E-20-8-1-N/librephotos-backend-local.git .
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Install necessary packages
RUN apt-get update && apt-get install -y \
    nano \
    vim

RUN if [ "$DEBUG" = 1 ] ; then \
    pip install --break-system-packages -r requirements.dev.txt; \
    fi
RUN if [ "$MLVALIDATION" = 1 ] ; then \
    apt-get update && apt-get install default-jre -y; \
    pip install --break-system-packages -r requirements.mlval.txt; \
    fi
EXPOSE 8001

COPY --chmod=755 entrypoint.sh /entrypoint.sh
CMD ["/entrypoint.sh"]