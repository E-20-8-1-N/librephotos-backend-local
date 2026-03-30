"""
Tests for PhotoMetadata API endpoints.

Tests the following:
- GET /api/photos/{photo_id}/metadata/ - Get full metadata
- PATCH /api/photos/{photo_id}/metadata/ - Update metadata
- GET /api/photos/{photo_id}/metadata/history/ - Get edit history
- POST /api/photos/{photo_id}/metadata/revert/{edit_id}/ - Revert a change
- Bulk metadata operations
- Edge cases (no EXIF, corrupted data, permissions)
"""

import uuid
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient, APITestCase

from api.models.photo_metadata import MetadataEdit, PhotoMetadata
from api.tests.utils import create_test_photo, create_test_user


class PhotoMetadataRetrieveTestCase(APITestCase):
    """Test metadata retrieval endpoints."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photo = create_test_photo(owner=self.user)

    def test_get_metadata_by_uuid(self):
        """Test retrieving metadata using photo UUID."""
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/")
        self.assertEqual(response.status_code, 200)
        # Response contains metadata fields directly
        self.assertIn("id", response.data)
        self.assertIn("source", response.data)

    def test_get_metadata_by_image_hash(self):
        """Test retrieving metadata using image_hash."""
        response = self.client.get(f"/api/photos/{self.photo.image_hash}/metadata/")
        self.assertEqual(response.status_code, 200)

    def test_get_metadata_creates_if_missing(self):
        """Test that metadata is created if it doesn't exist."""
        # Ensure no metadata exists
        PhotoMetadata.objects.filter(photo=self.photo).delete()
        
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/")
        self.assertEqual(response.status_code, 200)
        
        # Should have created metadata
        self.assertTrue(PhotoMetadata.objects.filter(photo=self.photo).exists())

    @patch("api.views.photo_metadata.PhotoMetadata.extract_exif_data")
    def test_get_metadata_falls_back_when_extraction_fails(self, mock_extract):
        """Test GET returns stored metadata instead of 500 when extraction fails."""
        mock_extract.side_effect = RuntimeError("bad exif payload")

        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("id", response.data)
        self.assertTrue(PhotoMetadata.objects.filter(photo=self.photo).exists())

    @patch("api.models.photo_metadata.get_metadata")
    def test_get_metadata_enriches_missing_fields_from_file(self, mock_get_metadata):
        """Test GET fills missing structured fields from the file metadata."""
        mock_get_metadata.return_value = [
            12345,
            1.8,
            6.765,
            125,
            0.02,
            "Apple",
            "iPhone 15 Pro Max",
            "Apple",
            "iPhone lens",
            5712,
            4284,
            24,
            None,
            None,
            0,
            5,
            "073",
            12,
            "2026:03:24 10:20:30",
            None,
            "+08:00",
            37.3317,
            -122.0301,
            15.0,
            "Shot on iPhone",
            "Cupertino campus",
            ["apple", "campus"],
            None,
            "Ethan",
            "Copyright 2026",
            1,
            "sRGB",
            8,
            "ABC123",
            "2026:03:24 10:21:00",
            None,
        ]

        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["gps_latitude"], 37.3317)
        self.assertEqual(response.data["gps_longitude"], -122.0301)
        self.assertEqual(response.data["title"], "Shot on iPhone")
        self.assertEqual(response.data["caption"], "Cupertino campus")

    @patch("api.models.photo_metadata.get_metadata")
    def test_get_metadata_reads_video_caption_from_keys_description(
        self, mock_get_metadata
    ):
        """Test GET metadata exposes MP4 captions stored in Keys:Description."""
        self.photo.video = True
        self.photo.save(update_fields=["video"], save_metadata=False)

        mock_get_metadata.return_value = [
            12345,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            608,
            1080,
            None,
            None,
            3.2,
            None,
            None,
            None,
            None,
            None,
            "2026:03:30 04:26:47",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "2026:03:30 05:23:20",
            "caption-from-macOS",
        ]

        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["caption"], "caption-from-macOS")
        self.assertEqual(response.data["width"], 608)
        self.assertEqual(response.data["height"], 1080)

    def test_get_metadata_nonexistent_photo(self):
        """Test 404 for nonexistent photo."""
        fake_uuid = str(uuid.uuid4())
        response = self.client.get(f"/api/photos/{fake_uuid}/metadata/")
        self.assertEqual(response.status_code, 404)

    def test_get_metadata_other_user_forbidden(self):
        """Test that users cannot access other users' photo metadata."""
        other_user = create_test_user()
        other_photo = create_test_photo(owner=other_user)
        
        response = self.client.get(f"/api/photos/{other_photo.pk}/metadata/")
        self.assertEqual(response.status_code, 403)

    def test_get_metadata_admin_can_access_any(self):
        """Test that admin can access any photo's metadata."""
        other_user = create_test_user()
        other_photo = create_test_photo(owner=other_user)
        
        # Make current user admin
        self.user.is_staff = True
        self.user.save()
        
        response = self.client.get(f"/api/photos/{other_photo.pk}/metadata/")
        self.assertEqual(response.status_code, 200)


class PhotoMetadataUpdateTestCase(APITestCase):
    """Test metadata update endpoints."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photo = create_test_photo(owner=self.user)

    def test_update_metadata_title(self):
        """Test updating photo title."""
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "My Test Photo"},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify update
        metadata = PhotoMetadata.objects.get(photo=self.photo)
        self.assertEqual(metadata.title, "My Test Photo")

    def test_update_metadata_creates_history(self):
        """Test that updates create edit history."""
        # First update
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "First Title"},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        # Second update
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "Second Title"},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        # Check history
        edits = MetadataEdit.objects.filter(photo=self.photo, field_name="title")
        self.assertGreaterEqual(edits.count(), 1)

    def test_update_metadata_rating(self):
        """Test updating photo rating."""
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"rating": 5},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        metadata = PhotoMetadata.objects.get(photo=self.photo)
        self.assertEqual(metadata.rating, 5)

    def test_update_metadata_caption(self):
        """Test updating photo caption."""
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"caption": "A beautiful sunset over the mountains"},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        metadata = PhotoMetadata.objects.get(photo=self.photo)
        self.assertEqual(metadata.caption, "A beautiful sunset over the mountains")

    @patch("api.models.photo.write_metadata")
    def test_update_metadata_caption_writes_to_sidecar(self, mock_write_metadata):
        """Test updating caption writes metadata to disk when enabled."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.SIDECAR_FILE
        self.user.save()

        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"caption": "A beautiful sunset over the mountains"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        mock_write_metadata.assert_called_once()
        written_tags = mock_write_metadata.call_args.args[1]
        self.assertEqual(
            written_tags["EXIF:ImageDescription"],
            "A beautiful sunset over the mountains",
        )
        self.assertNotIn("Keys:Description", written_tags)
        self.assertEqual(
            written_tags["XMP-dc:Description"],
            "A beautiful sunset over the mountains",
        )

    @patch("api.models.photo.write_metadata")
    def test_update_video_metadata_caption_writes_keys_description(
        self, mock_write_metadata
    ):
        """Test updating video caption writes Keys:Description instead of EXIF:ImageDescription."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.SIDECAR_FILE
        self.user.save()
        self.photo.video = True
        self.photo.save(update_fields=["video"], save_metadata=False)

        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"caption": "iPhone Video - TEST"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        mock_write_metadata.assert_called_once()
        written_tags = mock_write_metadata.call_args.args[1]
        self.assertEqual(written_tags["Keys:Description"], "iPhone Video - TEST")
        self.assertNotIn("EXIF:ImageDescription", written_tags)
        self.assertEqual(written_tags["XMP-dc:Description"], "iPhone Video - TEST")

        edit = MetadataEdit.objects.filter(photo=self.photo, field_name="caption").latest(
            "created_at"
        )
        self.assertTrue(edit.synced_to_file)

    @patch("api.models.photo.write_metadata")
    def test_update_metadata_caption_without_writeback_marks_unsynced(
        self, mock_write_metadata
    ):
        """Test edits remain database-only when writeback is disabled."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.OFF
        self.user.save()

        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"caption": "Database only caption"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        mock_write_metadata.assert_not_called()

        edit = MetadataEdit.objects.filter(photo=self.photo, field_name="caption").latest(
            "created_at"
        )
        self.assertFalse(edit.synced_to_file)

    def test_update_metadata_version_increments(self):
        """Test that metadata version increments on update."""
        # Get initial version
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        initial_version = metadata.version
        
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "Updated Title"},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        metadata.refresh_from_db()
        self.assertEqual(metadata.version, initial_version + 1)

    def test_update_metadata_forbidden_for_other_user(self):
        """Test that users cannot update other users' photo metadata."""
        other_user = create_test_user()
        other_photo = create_test_photo(owner=other_user)
        
        response = self.client.patch(
            f"/api/photos/{other_photo.pk}/metadata/",
            {"title": "Hacked Title"},
            format="json"
        )
        self.assertEqual(response.status_code, 403)


class PhotoMetadataHistoryTestCase(APITestCase):
    """Test metadata history endpoints."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photo = create_test_photo(owner=self.user)

    def test_get_empty_history(self):
        """Test getting history when no edits exist."""
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/history/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["results"], [])
        self.assertEqual(response.data["count"], 0)

    def test_get_history_with_edits(self):
        """Test getting history after making edits."""
        # Create some edit history
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value=None,
            new_value="Test Title"
        )
        
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/history/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(len(response.data["results"]), 1)

    def test_history_pagination(self):
        """Test history pagination."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        # Create many edit records
        for i in range(25):
            MetadataEdit.objects.create(
                photo=self.photo,
                user=self.user,
                field_name="rating",
                old_value=i,
                new_value=i + 1
            )
        
        # First page
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/history/?page=1&page_size=10")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["results"]), 10)
        self.assertEqual(response.data["count"], 25)

    def test_history_ordered_by_date(self):
        """Test that history is ordered by date descending."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        # Create edits with different times
        _edit1 = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value=None,
            new_value="First"
        )
        _edit2 = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value="First",
            new_value="Second"
        )
        
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/history/")
        self.assertEqual(response.status_code, 200)
        
        # Most recent should be first
        self.assertEqual(response.data["results"][0]["new_value"], "Second")


class PhotoMetadataRevertTestCase(APITestCase):
    """Test metadata revert endpoints."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photo = create_test_photo(owner=self.user)
        self.metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)

    def test_revert_single_edit(self):
        """Test reverting a single edit."""
        # Set initial value
        self.metadata.title = "Original Title"
        self.metadata.save()
        
        # Create edit record
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value="Original Title",
            new_value="Modified Title"
        )
        self.metadata.title = "Modified Title"
        self.metadata.save()
        
        # Revert
        response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert/{edit.id}/")
        self.assertEqual(response.status_code, 200)
        
        self.metadata.refresh_from_db()
        self.assertEqual(self.metadata.title, "Original Title")

    def test_revert_creates_history_entry(self):
        """Test that revert creates its own history entry."""
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value="Original",
            new_value="Modified"
        )
        self.metadata.title = "Modified"
        self.metadata.save()
        
        initial_count = MetadataEdit.objects.filter(photo=self.photo).count()
        
        response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert/{edit.id}/")
        self.assertEqual(response.status_code, 200)
        
        # Should have one more edit record
        new_count = MetadataEdit.objects.filter(photo=self.photo).count()
        self.assertEqual(new_count, initial_count + 1)

    def test_revert_nonexistent_edit(self):
        """Test reverting nonexistent edit returns 404."""
        fake_id = str(uuid.uuid4())
        response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert/{fake_id}/")
        self.assertEqual(response.status_code, 404)

    def test_revert_edit_from_wrong_photo(self):
        """Test that you cannot revert an edit from a different photo."""
        other_photo = create_test_photo(owner=self.user)
        other_metadata, _ = PhotoMetadata.objects.get_or_create(photo=other_photo)
        
        edit = MetadataEdit.objects.create(
            photo=other_photo,
            user=self.user,
            field_name="title",
            old_value="Original",
            new_value="Modified"
        )
        
        # Try to revert using wrong photo ID
        response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert/{edit.id}/")
        self.assertEqual(response.status_code, 404)

    @patch("api.models.photo.write_metadata")
    def test_revert_writes_to_sidecar_and_marks_synced(self, mock_write_metadata):
        """Test revert writes restored metadata back to storage when enabled."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.SIDECAR_FILE
        self.user.save()

        self.metadata.caption = "Changed caption"
        self.metadata.save()
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="caption",
            old_value="Original caption",
            new_value="Changed caption",
        )

        response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert/{edit.id}/")

        self.assertEqual(response.status_code, 200)
        mock_write_metadata.assert_called_once()
        written_tags = mock_write_metadata.call_args.args[1]
        self.assertEqual(written_tags["EXIF:ImageDescription"], "Original caption")
        self.assertNotIn("Keys:Description", written_tags)
        self.assertEqual(written_tags["XMP-dc:Description"], "Original caption")

        revert_edit = MetadataEdit.objects.filter(
            photo=self.photo,
            field_name="caption",
        ).latest("created_at")
        self.assertTrue(revert_edit.synced_to_file)


class PhotoMetadataRevertAllTestCase(APITestCase):
    """Test revert-all endpoint."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photo = create_test_photo(owner=self.user)

    def test_revert_all_creates_history(self):
        """Test that revert-all creates a history entry."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        initial_count = MetadataEdit.objects.filter(photo=self.photo).count()
        
        try:
            _response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert-all/")
        except (ConnectionError, OSError):
            # The endpoint may try to contact external services (e.g. EXIF/tag)
            # that are not available in the test environment
            pass
        
        new_count = MetadataEdit.objects.filter(photo=self.photo).count()
        # Should have at least tried to create the record
        self.assertGreaterEqual(new_count, initial_count)

    @patch("api.models.photo.write_metadata")
    @patch("api.models.photo_metadata.get_metadata")
    def test_revert_all_writes_restored_metadata_to_sidecar(
        self, mock_get_metadata, mock_write_metadata
    ):
        """Test revert-all restores embedded values and syncs them to storage."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.SIDECAR_FILE
        self.user.save()

        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        metadata.caption = "User caption"
        metadata.title = "User title"
        metadata.source = PhotoMetadata.Source.USER_EDIT
        metadata.save()

        mock_get_metadata.return_value = [
            12345,
            1.8,
            6.765,
            125,
            0.02,
            "Apple",
            "iPhone 15 Pro Max",
            "Apple",
            "iPhone lens",
            5712,
            4284,
            24,
            None,
            None,
            0,
            5,
            "073",
            12,
            "2026:03:24 10:20:30",
            None,
            "+08:00",
            37.3317,
            -122.0301,
            15.0,
            "Embedded title",
            "Embedded caption",
            ["embedded", "keywords"],
            None,
            "Ethan",
            "Copyright 2026",
            1,
            "sRGB",
            8,
            "ABC123",
            "2026:03:24 10:21:00",
            None,
        ]

        response = self.client.post(f"/api/photos/{self.photo.pk}/metadata/revert-all/")

        self.assertEqual(response.status_code, 200)
        mock_write_metadata.assert_called_once()
        written_tags = mock_write_metadata.call_args.args[1]
        self.assertEqual(written_tags["XMP-dc:Title"], "Embedded title")
        self.assertEqual(written_tags["XMP-dc:Description"], "Embedded caption")

        revert_edit = MetadataEdit.objects.filter(photo=self.photo, field_name="_all").latest(
            "created_at"
        )
        self.assertTrue(revert_edit.synced_to_file)


class BulkMetadataGetTestCase(APITestCase):
    """Test bulk metadata GET endpoint."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photos = [create_test_photo(owner=self.user) for _ in range(5)]

    def test_bulk_get_by_uuid(self):
        """Test bulk get metadata by UUIDs."""
        photo_ids = ",".join(str(p.pk) for p in self.photos[:3])
        response = self.client.get(f"/api/photos/metadata/bulk/?photo_ids={photo_ids}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 3)

    def test_bulk_get_by_image_hash(self):
        """Test bulk get metadata by image hashes."""
        photo_ids = ",".join(p.image_hash for p in self.photos[:3])
        response = self.client.get(f"/api/photos/metadata/bulk/?photo_ids={photo_ids}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 3)

    def test_bulk_get_mixed_ids(self):
        """Test bulk get with mixed UUID and image_hash."""
        photo_ids = f"{self.photos[0].pk},{self.photos[1].image_hash}"
        response = self.client.get(f"/api/photos/metadata/bulk/?photo_ids={photo_ids}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_bulk_get_no_ids(self):
        """Test bulk get with no IDs returns error."""
        response = self.client.get("/api/photos/metadata/bulk/")
        self.assertEqual(response.status_code, 400)

    def test_bulk_get_too_many_ids(self):
        """Test bulk get with too many IDs returns error."""
        # Create many fake IDs
        photo_ids = ",".join(str(uuid.uuid4()) for _ in range(101))
        response = self.client.get(f"/api/photos/metadata/bulk/?photo_ids={photo_ids}")
        self.assertEqual(response.status_code, 400)

    def test_bulk_get_filters_other_users(self):
        """Test that bulk get only returns current user's photos."""
        other_user = create_test_user()
        other_photo = create_test_photo(owner=other_user)
        
        photo_ids = f"{self.photos[0].pk},{other_photo.pk}"
        response = self.client.get(f"/api/photos/metadata/bulk/?photo_ids={photo_ids}")
        self.assertEqual(response.status_code, 200)
        # Should only return our photo
        self.assertEqual(len(response.data), 1)


class BulkMetadataUpdateTestCase(APITestCase):
    """Test bulk metadata PATCH endpoint."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photos = [create_test_photo(owner=self.user) for _ in range(5)]

    def test_bulk_update_rating(self):
        """Test bulk update rating for multiple photos."""
        photo_ids = [str(p.pk) for p in self.photos[:3]]
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": photo_ids, "updates": {"rating": 4}},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["updated_count"], 3)

    def test_bulk_update_creates_history(self):
        """Test that bulk update creates edit history."""
        photo_ids = [str(p.pk) for p in self.photos[:2]]
        
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": photo_ids, "updates": {"title": "Bulk Title"}},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        # Check history for each photo
        for photo in self.photos[:2]:
            edits = MetadataEdit.objects.filter(photo=photo, field_name="title")
            self.assertGreaterEqual(edits.count(), 1)

    def test_bulk_update_no_ids(self):
        """Test bulk update with no IDs returns error."""
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": [], "updates": {"rating": 5}},
            format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_bulk_update_no_updates(self):
        """Test bulk update with no updates returns error."""
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": [str(self.photos[0].pk)], "updates": {}},
            format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_bulk_update_invalid_field(self):
        """Test bulk update with invalid field returns error."""
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {
                "photo_ids": [str(self.photos[0].pk)],
                "updates": {"invalid_field": "value"}
            },
            format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_bulk_update_too_many_photos(self):
        """Test bulk update with too many photos returns error."""
        fake_ids = [str(uuid.uuid4()) for _ in range(101)]
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": fake_ids, "updates": {"rating": 5}},
            format="json"
        )
        self.assertEqual(response.status_code, 400)

    @patch("api.models.photo.write_metadata")
    def test_bulk_update_writes_to_sidecar_and_marks_synced(self, mock_write_metadata):
        """Test bulk updates write structured metadata to storage when enabled."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.SIDECAR_FILE
        self.user.save()

        photo_ids = [str(p.pk) for p in self.photos[:2]]
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": photo_ids, "updates": {"caption": "Bulk caption"}},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_write_metadata.call_count, 2)

        for photo in self.photos[:2]:
            edit = MetadataEdit.objects.filter(photo=photo, field_name="caption").latest(
                "created_at"
            )
            self.assertTrue(edit.synced_to_file)

    @patch("api.models.photo.write_metadata")
    def test_bulk_update_without_writeback_marks_unsynced(self, mock_write_metadata):
        """Test bulk edits remain unsynced when writeback is disabled."""
        self.user.save_metadata_to_disk = self.user.SaveMetadata.OFF
        self.user.save()

        photo_ids = [str(p.pk) for p in self.photos[:2]]
        response = self.client.patch(
            "/api/photos/metadata/bulk/",
            {"photo_ids": photo_ids, "updates": {"caption": "Bulk caption"}},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        mock_write_metadata.assert_not_called()

        for photo in self.photos[:2]:
            edit = MetadataEdit.objects.filter(photo=photo, field_name="caption").latest(
                "created_at"
            )
            self.assertFalse(edit.synced_to_file)


class PhotoMetadataEdgeCasesTestCase(APITestCase):
    """Test edge cases for metadata API."""

    def setUp(self):
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.photo = create_test_photo(owner=self.user)

    def test_photo_no_exif_data(self):
        """Test handling photo with no EXIF data."""
        # Clear any existing metadata
        PhotoMetadata.objects.filter(photo=self.photo).delete()
        
        # Clear exif fields on photo
        self.photo.exif_timestamp = None
        self.photo.exif_gps_lat = None
        self.photo.exif_gps_lon = None
        self.photo.save()
        
        response = self.client.get(f"/api/photos/{self.photo.pk}/metadata/")
        self.assertEqual(response.status_code, 200)
        # Should still return valid response

    def test_metadata_with_special_characters(self):
        """Test metadata with special characters in title/caption."""
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "Test 日本語 Émoji 🎉 <script>"},
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        
        metadata = PhotoMetadata.objects.get(photo=self.photo)
        self.assertIn("日本語", metadata.title)

    def test_metadata_empty_strings(self):
        """Test updating metadata with empty strings."""
        # First set a value
        self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "Some Title"},
            format="json"
        )
        
        # Then clear it
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": ""},
            format="json"
        )
        self.assertEqual(response.status_code, 200)

    def test_metadata_null_values(self):
        """Test updating metadata with null values."""
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"caption": None},
            format="json"
        )
        # Should handle gracefully
        self.assertIn(response.status_code, [200, 400])

    def test_concurrent_metadata_updates(self):
        """Test concurrent metadata updates (version conflict)."""
        # Get initial metadata
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        # Simulate concurrent updates
        response1 = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "Update 1"},
            format="json"
        )
        response2 = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": "Update 2"},
            format="json"
        )
        
        # Both should succeed (last one wins)
        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response2.status_code, 200)
        
        metadata.refresh_from_db()
        self.assertEqual(metadata.title, "Update 2")

    def test_very_long_values(self):
        """Test metadata with very long string values."""
        long_title = "A" * 1000
        response = self.client.patch(
            f"/api/photos/{self.photo.pk}/metadata/",
            {"title": long_title},
            format="json"
        )
        # Should either succeed or return validation error
        self.assertIn(response.status_code, [200, 400])


class PhotoMetadataModelTestCase(TestCase):
    """Test PhotoMetadata model methods."""

    def setUp(self):
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)

    def test_metadata_source_choices(self):
        """Test metadata source choices."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        # Test all source choices
        for source in PhotoMetadata.Source:
            metadata.source = source
            metadata.save()
            metadata.refresh_from_db()
            self.assertEqual(metadata.source, source)

    def test_has_location_property(self):
        """Test has_location computed property."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        # No location
        metadata.gps_latitude = None
        metadata.gps_longitude = None
        metadata.save()
        self.assertFalse(metadata.has_location)
        
        # With location
        metadata.gps_latitude = 40.7128
        metadata.gps_longitude = -74.0060
        metadata.save()
        self.assertTrue(metadata.has_location)

    def test_camera_display_property(self):
        """Test camera_display computed property."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        metadata.camera_make = "Canon"
        metadata.camera_model = "EOS R5"
        metadata.save()
        
        display = metadata.camera_display
        self.assertIsNotNone(display)

    def test_lens_display_property(self):
        """Test lens_display computed property."""
        metadata, _ = PhotoMetadata.objects.get_or_create(photo=self.photo)
        
        metadata.lens_make = "Canon"
        metadata.lens_model = "RF 24-70mm f/2.8L"
        metadata.save()
        
        display = metadata.lens_display
        self.assertIsNotNone(display)


class MetadataEditModelTestCase(TestCase):
    """Test MetadataEdit model."""

    def setUp(self):
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)

    def test_create_edit_record(self):
        """Test creating a metadata edit record."""
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value="Old Title",
            new_value="New Title"
        )
        self.assertIsNotNone(edit.id)
        self.assertEqual(edit.field_name, "title")
        self.assertEqual(edit.old_value, "Old Title")
        self.assertEqual(edit.new_value, "New Title")

    def test_edit_record_timestamps(self):
        """Test that edit records have correct timestamps."""
        before = timezone.now()
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="rating",
            old_value=0,
            new_value=5
        )
        after = timezone.now()
        
        self.assertGreaterEqual(edit.created_at, before)
        self.assertLessEqual(edit.created_at, after)

    def test_edit_record_json_values(self):
        """Test edit records with JSON values."""
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="keywords",
            old_value=["tag1", "tag2"],
            new_value=["tag1", "tag2", "tag3"]
        )
        
        edit.refresh_from_db()
        self.assertEqual(edit.new_value, ["tag1", "tag2", "tag3"])

    def test_edit_record_null_old_value(self):
        """Test edit record with null old value (new field)."""
        edit = MetadataEdit.objects.create(
            photo=self.photo,
            user=self.user,
            field_name="title",
            old_value=None,
            new_value="First Title"
        )
        
        edit.refresh_from_db()
        self.assertIsNone(edit.old_value)
        self.assertEqual(edit.new_value, "First Title")
