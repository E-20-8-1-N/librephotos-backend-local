"""Home-folder provisioning and immediate scans for system accounts."""

import logging
import os
from functools import partial

from django.conf import settings
from django.db import transaction
from django.db.models.signals import post_save, pre_save
from rest_framework.exceptions import NotFound, PermissionDenied, ValidationError

from api.accounts.directories import comparable_path
from api.directory_watcher import scan_photos
from api.drf.exceptions import JobAlreadyRunning
from api.jobs.queue import JobStep, enqueue_chain
from api.models import LongRunningJob, User
from api.photos.paths import is_valid_path
from api.runtime.ml_models import missing_models_prerequisites

logger = logging.getLogger(__name__)


def user_from_path(raw_path: str, must_exist: bool = True) -> tuple[User, str]:
    """Resolve the direct child username, preserving the upstream path checks."""
    root = comparable_path(settings.USER_HOME_DIRECTORY_PREFIX)
    path = comparable_path(raw_path)
    if not is_valid_path(path, root) or path == root:
        raise PermissionDenied("Path must be inside the configured user-home root")
    username = os.path.relpath(path, root).split(os.sep)[0]
    user = User.objects.filter(username=username).first()
    if user is None:
        raise NotFound("User not found")
    home = comparable_path(os.path.join(root, username))
    if not user.scan_directory or comparable_path(user.scan_directory) != home:
        raise PermissionDenied(
            "User scan directory does not match their home directory"
        )
    if must_exist and not os.path.exists(path):
        raise ValidationError({"path": "Path does not exist"})
    return user, path


def start_home_scan(user: User) -> LongRunningJob:
    [job] = enqueue_chain(
        user,
        [
            JobStep(
                LongRunningJob.JOB_SCAN_PHOTOS,
                scan_photos,
                (user, False),
                {
                    "scan_directory": user.scan_directory,
                    "force_im2txt": True,
                    "inline_image_processing": True,
                },
            )
        ],
        dispatch="thread",
        prerequisites=missing_models_prerequisites(user),
    )
    return job


def assign_home(sender: type[User], instance: User, **kwargs: object) -> None:
    root = settings.USER_HOME_DIRECTORY_PREFIX
    if root and not instance.scan_directory and instance.is_active:
        candidate = os.path.abspath(os.path.join(root, instance.username))
        if os.path.dirname(candidate) == os.path.abspath(root):
            instance.scan_directory = candidate


def scan_new_home(
    sender: type[User], instance: User, created: bool, **kwargs: object
) -> None:
    if (
        not settings.USER_HOME_DIRECTORY_PREFIX
        or not created
        or not instance.is_active
        or not os.path.isdir(instance.scan_directory)
    ):
        return
    transaction.on_commit(partial(_scan_created_user, instance.pk))


def _scan_created_user(user_id: int) -> None:
    user = User.objects.filter(pk=user_id).first()
    if user is None:
        return
    try:
        start_home_scan(user)
    except JobAlreadyRunning:
        return
    except Exception:
        logger.exception("Could not start the home scan for user %s", user_id)


def register() -> None:
    pre_save.connect(
        assign_home, sender=User, dispatch_uid="api.accounts.home_scan.assign_home"
    )
    post_save.connect(
        scan_new_home, sender=User, dispatch_uid="api.accounts.home_scan.scan_new_home"
    )
