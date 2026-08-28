"""
File and Photo creation handlers.

This module contains functions for creating File records and grouping
them into Photo objects.
"""

import datetime
import os

import pytz
from django.conf import settings
from django.db import transaction
from django.db.models import Q

from api import util
from api.models import File, Photo, Thumbnail
from api.models.file import calculate_hash, is_metadata, is_raw, is_valid_media, is_video
from api.models.photo_search import PhotoSearch
from api.perceptual_hash import calculate_hash_from_thumbnail
from api.stacks.live_photo import has_embedded_motion_video, extract_embedded_motion_video

from api.directory_watcher.file_grouping import (
    FILE_TYPE_PRIORITY,
    find_matching_jpeg_photo,
    find_matching_image_for_video,
    select_main_file,
)
from api.directory_watcher.utils import update_scan_counter


def create_file_record(user, path) -> File | None:
    """
    Phase 1: Create a File record for a path without creating/grouping Photos.
    
    This is the first phase of the two-phase scan architecture:
    - Phase 1: Create File records for all discovered files (this function)
    - Phase 2: Group files into Photos by (directory, basename)
    
    This separation eliminates race conditions where concurrent processing
    of RAW and JPEG files could create separate Photos instead of grouping them.
    
    Args:
        user: The owner of the file
        path: The file path
        
    Returns:
        File object if created/found, None if invalid media
    """
    if not is_valid_media(path=path, user=user):
        return None
    
    hash_value = calculate_hash(user, path)
    
    # Skip if this is embedded media (already attached to another file)
    if File.embedded_media.through.objects.filter(Q(to_file_id=hash_value)).exists():
        util.logger.warning(f"embedded content file found {path}")
        return None
    
    # File.create handles concurrent path and content-hash conflicts.
    file = File.create(path, user)
    return file


def group_files_into_photo(user, files: list[File], job_id) -> Photo | None:
    """
    Phase 2: Group a list of related files into a single Photo.
    
    Creates a new Photo with the given files as variants, selecting the
    best file as main_file based on type priority (IMAGE > VIDEO > RAW > METADATA).
    
    This function should be called with all files that share the same
    (directory, basename) - e.g., IMG_001.jpg, IMG_001.CR2, IMG_001.xmp.
    
    Args:
        user: The owner of the photo
        files: List of File objects to group (must not be empty)
        job_id: Job ID for logging
        
    Returns:
        The created or existing Photo, or None if no valid files
    """
    if not files:
        return None

    file_hashes = sorted({file.hash for file in files})

    # The backend scan processes groups concurrently. Locking the canonical
    # File rows serializes imports that resolve to the same hash or path, so a
    # second worker sees the Photo created by the first worker.
    with transaction.atomic():
        files = list(
            File.objects.select_for_update()
            .filter(hash__in=file_hashes)
            .order_by("hash")
        )

        # Filter out metadata files for main photo creation - they're sidecars
        non_metadata_files = [
            file for file in files if file.type != File.METADATA_FILE
        ]

        if not non_metadata_files:
            # Only metadata files - no photo to create
            util.logger.warning(
                f"job {job_id}: Only metadata files in group, skipping"
            )
            return None

        # Select main file based on priority
        main_file = select_main_file(non_metadata_files)
        if not main_file:
            return None

        file_paths = [file.path for file in files]
        existing_photo = (
            Photo.objects.filter(owner=user)
            .filter(
                Q(image_hash__in=file_hashes)
                | Q(files__hash__in=file_hashes)
                | Q(files__path__in=file_paths)
                | Q(main_file__hash__in=file_hashes)
                | Q(main_file__path__in=file_paths)
            )
            .distinct()
            .order_by("removed", "in_trashcan", "added_on", "id")
            .first()
        )

        if existing_photo:
            existing_hashes = set(
                existing_photo.files.values_list("hash", flat=True)
            )
            missing_files = [
                file for file in files if file.hash not in existing_hashes
            ]
            if missing_files:
                existing_photo.files.add(*missing_files)
                for file in missing_files:
                    util.logger.info(
                        f"job {job_id}: Attached file {file.path} to existing "
                        f"Photo {existing_photo.image_hash}"
                    )

            update_fields = []
            if existing_photo.main_file:
                current_priority = FILE_TYPE_PRIORITY.get(
                    existing_photo.main_file.type, 999
                )
                new_priority = FILE_TYPE_PRIORITY.get(main_file.type, 999)
                if new_priority < current_priority:
                    existing_photo.main_file = main_file
                    existing_photo.video = main_file.type == File.VIDEO
                    update_fields.extend(["main_file", "video"])
            else:
                existing_photo.main_file = main_file
                existing_photo.video = main_file.type == File.VIDEO
                update_fields.extend(["main_file", "video"])

            if existing_photo.removed:
                existing_photo.removed = False
                existing_photo.in_trashcan = False
                update_fields.extend(["removed", "in_trashcan"])

            if update_fields:
                existing_photo.save(update_fields=update_fields)

            util.logger.info(
                f"job {job_id}: Reused Photo {existing_photo.image_hash} for "
                f"{len(files)} file(s)"
            )
            return existing_photo

        photo = Photo(
            image_hash=main_file.hash,
            owner=user,
            added_on=datetime.datetime.now().replace(tzinfo=pytz.utc),
            geolocation_json={},
            video=main_file.type == File.VIDEO,
            main_file=main_file,
        )
        photo.save()
        photo.files.add(*files)
    
    # Handle embedded media (Google/Samsung Live Photos with embedded video)
    if settings.FEATURE_PROCESS_EMBEDDED_MEDIA and has_embedded_motion_video(
        main_file.path
    ):
        em_path = extract_embedded_motion_video(main_file.path, main_file.hash)
        if em_path:
            em_file = File.create(em_path, user)
            main_file.embedded_media.add(em_file)
            photo.files.add(em_file)
    
    util.logger.info(f"job {job_id}: Created Photo {photo.image_hash} with {len(files)} file(s)")
    return photo


def create_new_image(user, path) -> Photo | None:
    """
    Creates a new Photo object based on user input and file path.
    
    This is the legacy single-file creation function, kept for backwards
    compatibility with upload handling. For scan operations, use the
    two-phase approach (create_file_record + group_files_into_photo).

    Args:
        user: The owner of the photo.
        path: The file path of the image.

    Returns:
        The created or existing Photo object if successful, otherwise returns None.

    Note:
        This function implements file variant grouping (PhotoPrism-like):
        - RAW files are attached to existing JPEG Photos as file variants
        - Live Photo videos (.mov) are attached to existing image Photos as file variants
        - Other files create new Photo entities
    """
    if not is_valid_media(path=path, user=user):
        return None
    hash_value = calculate_hash(user, path)
    if File.embedded_media.through.objects.filter(Q(to_file_id=hash_value)).exists():
        util.logger.warning(f"embedded content file found {path}")
        return None

    # Handle metadata files (XMP sidecars)
    if is_metadata(path):
        photo_name = os.path.splitext(os.path.basename(path))[0]
        photo_dir = os.path.dirname(path)
        photo = Photo.objects.filter(
            Q(files__path__contains=photo_dir)
            & Q(files__path__contains=photo_name)
            & ~Q(files__path__contains=os.path.basename(path))
        ).first()

        if photo:
            file = File.create(path, user)
            photo.files.add(file)
            photo.save()
        else:
            util.logger.warning(f"no photo to metadata file found {path}")
        return None

    # === File Variant Handling (PhotoPrism-like model) ===
    
    # Handle RAW files: attach to existing JPEG Photo if found
    if is_raw(path):
        existing_photo = find_matching_jpeg_photo(path, user)
        if existing_photo:
            # Check if this RAW file is already attached
            if not existing_photo.files.filter(path=path).exists():
                raw_file = File.create(path, user)
                existing_photo.files.add(raw_file)
                existing_photo.save()
                util.logger.info(f"Attached RAW file {path} to existing Photo {existing_photo.image_hash}")
            return existing_photo
    
    # Handle Live Photo videos (.mov): attach to existing image Photo if found
    if is_video(path):
        existing_photo = find_matching_image_for_video(path, user)
        if existing_photo:
            # Check if this video is already attached
            if not existing_photo.files.filter(path=path).exists():
                video_file = File.create(path, user)
                existing_photo.files.add(video_file)
                existing_photo.video = False  # Keep photo as image (video is just a variant)
                existing_photo.save()
                util.logger.info(f"Attached Live Photo video {path} to existing Photo {existing_photo.image_hash}")
            return existing_photo

    # === Standard Photo Creation ===
    file = File.create(path, user)
    return group_files_into_photo(user, [file], "direct-import")


def handle_new_image(user, path, job_id, photo=None):
    """
    Handles the creation and all the processing of the photo needed for it to be displayed.

    Args:
        user: The owner of the photo.
        path: The file path of the image.
        job_id: The long-running job id, which gets updated when the task runs
        photo: An optional parameter, where you can input a photo instead of creating a new one. Used for uploading.

    Note:
        This function is used when uploading a picture, because rescanning does not perform machine learning tasks.
    """
    try:
        start = datetime.datetime.now()
        if photo is None:
            photo = create_new_image(user, path)
            elapsed = (datetime.datetime.now() - start).total_seconds()
            util.logger.info(f"job {job_id}: save image: {path}, elapsed: {elapsed}")
        if photo:
            _process_photo(photo, path, job_id, start)

    except Exception as e:
        try:
            util.logger.exception(
                f"job {job_id}: could not load image {path}. reason: {str(e)}"
            )
        except Exception:
            util.logger.exception(f"job {job_id}: could not load image {path}")
    finally:
        update_scan_counter(job_id)


def handle_file_group(user, file_paths: list[str], job_id):
    """
    Phase 2 handler: Process a group of related files into a single Photo.
    
    This is called after Phase 1 has created File records for all paths.
    Files are grouped by (directory, basename) so RAW+JPEG pairs are processed together.
    
    Args:
        user: The owner of the files
        file_paths: List of file paths that share the same (directory, basename)
        job_id: Job ID for logging and progress tracking
    """
    try:
        start = datetime.datetime.now()
        
        # Get or create File records for all paths
        files = []
        for path in file_paths:
            file = create_file_record(user, path)
            if file:
                files.append(file)
        
        if not files:
            util.logger.warning(f"job {job_id}: No valid files in group: {file_paths}")
            return
        
        # Group files into a Photo
        photo = group_files_into_photo(user, files, job_id)
        
        if not photo:
            util.logger.warning(f"job {job_id}: Could not create photo for files: {file_paths}")
            return
        
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(f"job {job_id}: created photo with {len(files)} files, elapsed: {elapsed}")
        
        # Process the photo (thumbnails, EXIF, etc.) using main_file
        if photo.main_file:
            _process_photo(photo, photo.main_file.path, job_id, start)

    except Exception as e:
        try:
            util.logger.exception(
                f"job {job_id}: could not process file group {file_paths}. reason: {str(e)}"
            )
        except Exception:
            util.logger.exception(f"job {job_id}: could not process file group")
    finally:
        update_scan_counter(job_id)


def _process_photo(photo: Photo, path: str, job_id, start: datetime.datetime):
    """
    Process a photo: generate thumbnails, extract EXIF, calculate hashes, etc.
    
    This is the common processing logic shared between handle_new_image and handle_file_group.
    
    Args:
        photo: The Photo object to process
        path: The main file path (for logging)
        job_id: Job ID for logging
        start: Start time for elapsed time calculation
    """
    util.logger.info(f"job {job_id}: handling image {path}")
    
    # Create or get thumbnail instance
    thumbnail, _ = Thumbnail.objects.get_or_create(photo=photo)
    try:
        thumbnail._generate_thumbnail()
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(
            f"job {job_id}: generate thumbnails: {path}, elapsed: {elapsed}"
        )
    except Exception as e:
        util.logger.error(f"job {job_id}: Failed to generate thumbnail for {path}: {e}")
    
    # Calculate Aspect Ratio
    try:
        thumbnail._calculate_aspect_ratio()
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(
            f"job {job_id}: calculate aspect ratio: {path}, elapsed: {elapsed}"
        )
    except Exception as e:
        util.logger.warning(f"job {job_id}: Failed to calculate aspect ratio for {path} (skipping): {e}")
    
    # Calculate perceptual hash for duplicate detection
    try:
        if thumbnail.thumbnail_big and os.path.exists(thumbnail.thumbnail_big.path):
            phash = calculate_hash_from_thumbnail(thumbnail.thumbnail_big.path)
            if phash:
                photo.perceptual_hash = phash
                photo.save(update_fields=["perceptual_hash"])
                elapsed = (datetime.datetime.now() - start).total_seconds()
                util.logger.info(
                    f"job {job_id}: calculate perceptual hash: {path}, elapsed: {elapsed}"
                )
    except Exception as e:
        util.logger.error(f"job {job_id}: Failed to calculate perceptual hash for {path}: {e}")
    
    # Extract EXIF Data
    try:
        from api.models.photo_metadata import PhotoMetadata
        PhotoMetadata.extract_exif_data(photo, commit=True)
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(
            f"job {job_id}: extract exif data: {path}, elapsed: {elapsed}"
        )
    except Exception as e:
        util.logger.error(f"job {job_id}: Failed to extract EXIF for {path}: {e}")

    # Extract Date/Time
    try:
        photo._extract_date_time_from_exif(True)
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(
            f"job {job_id}: extract date time: {path}, elapsed: {elapsed}"
        )
    except Exception as e:
        util.logger.error(f"job {job_id}: Failed to extract date/time for {path}: {e}")
    
    # Dominant Color
    try:
        thumbnail._get_dominant_color()
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(
            f"job {job_id}: get dominant color: {path}, elapsed: {elapsed}"
        )
    except Exception as e:
        util.logger.warning(f"job {job_id}: Failed to get dominant color for {path}: {e}")
    
    # Search Captions
    try:
        search_instance, created = PhotoSearch.objects.get_or_create(photo=photo)
        search_instance.recreate_search_captions()
        search_instance.save()
        elapsed = (datetime.datetime.now() - start).total_seconds()
        util.logger.info(
            f"job {job_id}: search caption recreated: {path}, elapsed: {elapsed}"
        )
    except Exception as e:
        util.logger.error(f"job {job_id}: Failed to recreate search captions for {path}: {e}")
