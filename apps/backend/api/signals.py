import os
import threading
import uuid

from django import db
from django.db.models.signals import post_save
from django.dispatch import receiver

from api import util
from api.directory_watcher import scan_photos
from api.ml_models import do_all_models_exist, download_models
from api.models.user import User


def _run_user_scan(user, job_id):
    """
    Run the scan orchestration for a single user directly in a background
    thread so that:
      - multiple users triggering a scan concurrently each start orchestrating
        immediately (no django-q worker slot required just to orchestrate),
      - a new user's scan does not have to wait for another user's scan_photos
        orchestrator to be dequeued before its own subtasks get queued.

    Heavy per-file work is still dispatched to django-q workers from inside
    scan_photos(), so worker-pool contention still applies to the actual
    processing — but queuing happens in parallel for each user.
    """
    try:
        if not do_all_models_exist():
            try:
                download_models(user)
            except Exception:
                util.logger.exception(
                    "download_models failed for user %s; aborting auto scan",
                    getattr(user, "username", user.pk),
                )
                return

        scan_photos(user, False, job_id, user.scan_directory, None, True)
    except Exception:
        util.logger.exception(
            "auto scan thread failed for user %s",
            getattr(user, "username", user.pk),
        )
    finally:
        # Ensure the thread doesn't leak a DB connection.
        db.connections.close_all()


@receiver(post_save, sender=User)
def auto_scan_new_user_directory(sender, instance, created, **kwargs):
    if not created or not instance.is_active:
        return

    if not instance.scan_directory or not os.path.exists(instance.scan_directory):
        return

    thread = threading.Thread(
        target=_run_user_scan,
        args=(instance, uuid.uuid4()),
        name=f"auto-scan-{instance.pk}",
        daemon=True,
    )
    thread.start()