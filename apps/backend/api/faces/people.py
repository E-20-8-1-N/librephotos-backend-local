"""Keep photo search in sync when a named person is edited or deleted."""

from django.db import transaction
from django.db.models import Prefetch, QuerySet

from api.faces.labels import recreate_search_captions
from api.models.face import Face
from api.models.person import Person
from api.models.photo import Photo
from api.models.user import User


def rename_person(person: Person, user: User, name: str) -> None:
    """Rename a person and rebuild the search text of their owner's photos."""
    with transaction.atomic():
        person.name = name
        person.save(update_fields=["name", "last_modified"])
        _refresh_search_captions(
            Photo.objects.owned_by(user).filter(faces__person=person).distinct()
        )


def delete_person(person: Person, user: User) -> None:
    """Delete a person and remove their face labels from photo search."""
    with transaction.atomic():
        # SET_NULL unlinks the faces on deletion, so remember the photos first.
        photo_ids = list(
            Photo.objects.owned_by(user)
            .filter(faces__person=person)
            .values_list("pk", flat=True)
            .distinct()
        )
        person.delete()
        # Load faces after the delete so their cached people are up to date.
        _refresh_search_captions(Photo.objects.owned_by(user).filter(pk__in=photo_ids))


def _refresh_search_captions(photos: QuerySet[Photo]) -> None:
    """Load every caption source in bulk before rebuilding the search rows."""
    recreate_search_captions(
        list(
            photos.select_related(
                "main_file", "metadata", "caption_instance"
            ).prefetch_related(
                "files",
                Prefetch("faces", queryset=Face.objects.select_related("person")),
            )
        )
    )
