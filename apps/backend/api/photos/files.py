"""The files behind a photo: detaching the missing ones and removing a photo
together with the files only it uses.

Moved out of ``Photo``; every function takes the photo as its first argument.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from django.db import transaction

from api.models.file import File
from api.video import transcode_cache

if TYPE_CHECKING:
    from api.models.photo import Photo

logger = logging.getLogger(__name__)


def missing_files(photo: Photo) -> list[File]:
    """The photo's files that are gone from disk.

    A file whose own path is gone but which has an exact copy on disk (a
    ``FileCopy``, #2417) is not missing: the copy takes the file over here,
    as the next scan would, so the photo is not flagged missing meanwhile.
    """
    return [file for file in photo.files.all() if not _on_disk(file)]


def missing_on_disk(photo: Photo) -> bool:
    """True when no media variant or main file survives, including exact copies.

    An XMP sidecar cannot keep a photo alive by itself. The main file may no
    longer be in ``files`` after a previous missing-file check detached it.
    """
    media = {
        file.pk: file for file in photo.files.all() if file.type != File.METADATA_FILE
    }
    if photo.main_file is not None and photo.main_file.type != File.METADATA_FILE:
        media.setdefault(photo.main_file.pk, photo.main_file)
    embedded = set(
        File.embedded_media.through.objects.filter(to_file_id__in=media).values_list(
            "to_file_id", flat=True
        )
    )
    media = {key: file for key, file in media.items() if key not in embedded}
    return not any(_on_disk(file) for file in media.values())


def delete_unreferenced_files(file_ids: Iterable[str]) -> int:
    """Remove detached File rows and unreferenced embedded previews.

    The caller selects the file ids through owned photos. A row another photo
    or parent file still uses is kept. Only derived embedded media is removed
    from disk; surviving originals and sidecars are never unlinked here.
    """
    ids = set(file_ids)
    embedded = set(
        File.embedded_media.through.objects.filter(from_file_id__in=ids).values_list(
            "to_file_id", flat=True
        )
    )
    ids.update(embedded)
    deleted = 0
    with transaction.atomic():
        # Parents go first; the next pass can then remove their orphaned children.
        for _ in range(2):
            used_as_embedded = File.embedded_media.through.objects.values_list(
                "to_file_id", flat=True
            )
            orphaned = list(
                File.objects.filter(
                    pk__in=ids, photo__isnull=True, main_photo__isnull=True
                )
                .exclude(pk__in=used_as_embedded)
                .distinct()
            )
            if not orphaned:
                break
            File.objects.filter(pk__in=[file.pk for file in orphaned]).delete()
            deleted += len(orphaned)
            for file in orphaned:
                if file.pk in embedded:
                    transaction.on_commit(partial(_remove_embedded_file, file.path))
                ids.discard(file.pk)
    return deleted


def _remove_embedded_file(path: str) -> None:
    if File.objects.filter(path=path).exists():
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        logger.warning("Could not remove an orphaned embedded preview")


def _on_disk(file: File) -> bool:
    """Whether ``file`` is on disk, at its own path or at a copy that takes it over."""
    if file.path and os.path.exists(file.path):
        return True
    for copy in file.copies.values_list("path", flat=True):
        if os.path.exists(copy):
            logger.info("a copy takes over the missing file %s", file.hash)
            file.take_over_copy(copy)
            return True
    return False


def detach_missing_files(photo: Photo, missing: list[File] | None = None) -> None:
    """Unlink files that are gone from disk and flag them missing.

    ``missing`` is the result of ``missing_files(photo)`` when the caller has
    already looked (the scan checks the whole library before it changes
    anything); without it the files are checked here.

    The scan runs this over every photo in the library, from rows it loaded
    in pages of thousands. Saving each one whole wrote those old values back
    over anything that changed in the meantime -- a rating given while the
    scan ran, a field another job had just filled in -- and bumped every
    ``last_modified``, so a removed photo never got old enough for
    ``cleanup_deleted_photos``. Only a photo that actually lost a file is
    touched, and only its ``last_modified``: the relation itself is already
    written by ``remove()``, and the bump is what tells the sync feed.
    """
    if missing is None:
        missing = missing_files(photo)
    for file in missing:
        photo.files.remove(file)
        file.missing = True
        file.save()
    if missing:
        photo.save(update_fields=["last_modified"])


def remove_photo(photo: Photo) -> None:
    """Delete the photo's unshared files from disk and mark it removed.

    Files another photo still uses are only unlinked. Stacks and duplicate
    groups left with one photo or none are dissolved.
    """
    # Store stack references before cleanup (ManyToMany)
    photo_stacks = list(photo.stacks.all())

    # Store duplicate group references before cleanup (ManyToMany)
    photo_duplicates = list(photo.duplicates.all())

    # Handle file cleanup - only delete files not shared with other Photos
    for file in photo.files.all():
        # Check if this file is used by other Photos (via files M2M or as main_file)
        other_photos_using_file = (
            file.photo_set.exclude(pk=photo.pk).exists()
            or file.main_photo.exclude(pk=photo.pk).exists()
        )

        if other_photos_using_file:
            # File is shared - just unlink from this photo, don't delete
            logger.info(
                "File %s is shared with other photos, unlinking only", file.path
            )
            photo.files.remove(file)
        else:
            # File is only used by this photo - safe to delete
            if os.path.isfile(file.path):
                logger.info("Removing photo %s", file.path)
                os.remove(file.path)
            file.delete()

    photo.files.set([])
    photo.main_file = None
    photo.removed = True

    # A cached transcode outlives the photo otherwise: it is named after the
    # image hash, which no longer belongs to anything, so nothing would ever
    # serve it and nothing would ever reclaim it until the cache filled up.
    transcode_cache.discard(photo.image_hash)

    # Clear all stack references from this photo (ManyToMany)
    photo.stacks.clear()

    # Clear all duplicate group references from this photo (ManyToMany)
    photo.duplicates.clear()

    photo.save()

    # Clean up stacks if they're now empty or have only one photo left
    for photo_stack in photo_stacks:
        remaining_photos = photo_stack.photos.filter(removed=False).count()
        if remaining_photos <= 1:
            # If 0 or 1 photos left, delete the stack (no longer a valid grouping)
            logger.info(
                "Deleting photo stack %s - only %s photos remaining",
                photo_stack.id,
                remaining_photos,
            )
            # Unlink remaining photos from stack first
            for remaining_photo in photo_stack.photos.all():
                remaining_photo.stacks.remove(photo_stack)
            photo_stack.delete()

    # Clean up duplicate groups if they're now empty or have only one photo left
    for duplicate in photo_duplicates:
        remaining_photos = duplicate.photos.filter(removed=False).count()
        if remaining_photos <= 1:
            # If 0 or 1 photos left, delete the duplicate group (no longer valid)
            logger.info(
                "Deleting duplicate group %s - only %s photos remaining",
                duplicate.id,
                remaining_photos,
            )
            # Unlink remaining photos from duplicate first
            for remaining_photo in duplicate.photos.all():
                remaining_photo.duplicates.remove(duplicate)
            duplicate.delete()


class Removal(NamedTuple):
    """What ``remove_photos`` did: how many photos went, which ones failed."""

    removed: int
    failed: list[Photo]


def remove_photos(
    photos: Iterable[Photo], progress: Callable[[int], None] | None = None
) -> Removal:
    """``remove_photo`` each photo, logging and skipping one that fails.

    One unreadable file or broken row must not stop the rest: Delete
    permanently, Empty trash and the trash retention all go through here.
    ``progress`` gets the number of photos handled so far after each one; an
    exception it raises (a job's cancel) stops the loop.
    """
    removed = 0
    failed: list[Photo] = []
    for done, photo in enumerate(photos, start=1):
        try:
            remove_photo(photo)
        except Exception:
            logger.exception(
                "Could not delete photo %s, skipping it.", photo.image_hash
            )
            failed.append(photo)
        else:
            removed += 1
        if progress is not None:
            progress(done)
    return Removal(removed, failed)
