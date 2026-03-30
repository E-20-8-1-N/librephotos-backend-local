from django.apps import AppConfig


class ApiConfig(AppConfig):
    name = "api"
    verbose_name = "LibrePhotos"

    def ready(self):
        from api import signals  # noqa: F401
