import os
import uuid

from django.db.models.signals import post_save
from django.dispatch import receiver
from django_q.tasks import Chain

from api.directory_watcher import scan_photos
from api.ml_models import do_all_models_exist, download_models
from api.models.user import User


@receiver(post_save, sender=User)
def auto_scan_new_user_directory(sender, instance, created, **kwargs):
    if not created or not instance.is_active:
        return

    if not instance.scan_directory or not os.path.exists(instance.scan_directory):
        return

    chain = Chain()
    if not do_all_models_exist():
        chain.append(download_models, instance)

    chain.append(scan_photos, instance, False, uuid.uuid4(), instance.scan_directory)
    chain.run()