import tempfile

from librephotos.logging_bootstrap import build_logging_config

from . import production as _production
from .production import *  # noqa

DEBUG = True

# The upstream suite covers the built-in model and explicit library assignment.
# Fork integration tests opt into the home root and external captioner.
USER_HOME_DIRECTORY_PREFIX = ""
CONSTANCE_CONFIG = {
    **_production.CONSTANCE_CONFIG,
    "CAPTIONING_MODEL": ("None", "Captioning model", "captioning_model"),
    "SCAN_SKIP_EXTENSIONS": ("", "File extensions excluded from scanning", str),
}

# The suite has to stay quiet, but it cannot lose the file handler:
# logging_bootstrap.get_file_handler resolves the live rotating handler out of
# this configuration by name, and a NullHandler root would leave it with nothing
# to find and the rotation tests with nothing to assert about. So keep the real handler, point it at a
# throwaway directory instead of the deployment's log directory, and silence it
# with the level rather than by removing it.
#
# LOGS_ROOT moves with the handler rather than staying on BASE_LOGS:
# ServerLogsView serves settings.LOGS_ROOT/LOG_FILENAME, so leaving the two
# pointing at different directories would let the suite pass while an admin
# downloads a file nothing writes to.
LOGS_ROOT = tempfile.mkdtemp(prefix="librephotos-test-logs-")
LOGGING = build_logging_config(
    logs_root=LOGS_ROOT,
    level="CRITICAL",
    to_console=False,
    # A developer's LOG_LEVELS must not switch loggers back on mid-suite.
    logger_levels={},
)

# The suite signs in and signs up from 127.0.0.1 many times a minute in each
# process, so the throttles get a rate it never reaches. The throttle tests
# (api/tests/users_and_auth/test_auth_throttling.py) set their own and clear
# the cache.
REST_FRAMEWORK = {
    **REST_FRAMEWORK,  # noqa: F405
    "DEFAULT_THROTTLE_RATES": {
        **DEFAULT_THROTTLE_RATES,  # noqa: F405
        "auth": "100000/minute",
        "token_refresh": "100000/minute",
    },
}
