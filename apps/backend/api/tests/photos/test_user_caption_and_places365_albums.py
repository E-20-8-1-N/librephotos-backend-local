"""Characterization tests for PhotoCaption.save_user_caption.

These tests pin the CURRENT observed behavior before refactoring. They
deliberately encode quirks (substring-based hashtag removal, silent
exception swallowing, etc.) rather than aspirational behavior.
"""

from unittest.mock import patch

from django.test import TestCase

from api.models import PhotoCaption
from api.models.album_thing import AlbumThing, get_album_thing
from api.tests.utils import create_test_photo, create_test_user


class SaveUserCaptionTest(TestCase):
    def setUp(self):
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)
        self.caption = PhotoCaption.objects.create(photo=self.photo)

    def _hashtag_albums(self):
        return set(
            AlbumThing.objects.filter(
                thing_type="hashtag_attribute", owner=self.user
            ).values_list("title", flat=True)
        )

    # ---------- thumbnail guard ----------

    def test_returns_false_when_thumbnail_big_missing(self):
        """No thumbnail_big -> early return False, captions_json untouched."""
        thumb = self.photo.thumbnail
        thumb.thumbnail_big = ""
        thumb.save()
        self.photo.refresh_from_db()

        result = self.caption.save_user_caption("hello", commit=True)

        self.assertFalse(result)
        self.assertIsNone(self.caption.captions_json)

    # ---------- happy path ----------

    def test_saves_caption_and_returns_true(self):
        result = self.caption.save_user_caption("My beautiful photo", commit=True)

        self.assertTrue(result)
        self.caption.refresh_from_db()
        self.assertEqual(
            self.caption.captions_json["user_caption"], "My beautiful photo"
        )

    def test_strips_start_end_markers_and_whitespace(self):
        result = self.caption.save_user_caption(
            "  <start> a dog on grass <end>  ", commit=False
        )

        self.assertTrue(result)
        self.assertEqual(self.caption.captions_json["user_caption"], "a dog on grass")

    def test_commit_false_does_not_persist_but_mutates_instance(self):
        result = self.caption.save_user_caption("not committed", commit=False)

        self.assertTrue(result)
        self.assertEqual(self.caption.captions_json["user_caption"], "not committed")
        fresh = PhotoCaption.objects.get(photo=self.photo)
        self.assertIsNone(fresh.captions_json)

    def test_preserves_other_caption_keys(self):
        self.caption.captions_json = {"im2txt": "a machine caption"}
        self.caption.save()

        self.caption.save_user_caption("human caption", commit=True)

        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["im2txt"], "a machine caption")
        self.assertEqual(self.caption.captions_json["user_caption"], "human caption")

    def test_empty_caption_is_saved_as_empty_string(self):
        result = self.caption.save_user_caption("   ", commit=True)

        self.assertTrue(result)
        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["user_caption"], "")

    def test_recreate_search_captions_is_called(self):
        with patch.object(PhotoCaption, "recreate_search_captions") as mock_recreate:
            self.caption.save_user_caption("something", commit=False)
        mock_recreate.assert_called_once_with()

    # ---------- hashtags ----------

    def test_hashtags_create_album_things(self):
        self.caption.save_user_caption("beach day #sun #sea", commit=True)

        self.assertEqual(self._hashtag_albums(), {"#sun", "#sea"})
        for title in ("#sun", "#sea"):
            album = AlbumThing.objects.get(
                title=title, thing_type="hashtag_attribute", owner=self.user
            )
            self.assertEqual(album.photos.count(), 1)

    def test_bare_hash_is_not_treated_as_hashtag(self):
        """A lone '#' (len == 1) is filtered out."""
        self.caption.save_user_caption("just a # here", commit=True)

        self.assertEqual(self._hashtag_albums(), set())

    def test_hashtag_must_start_the_word(self):
        """Mid-word hashes are not hashtags (split on whitespace, startswith)."""
        self.caption.save_user_caption("foo#bar baz", commit=True)

        self.assertEqual(self._hashtag_albums(), set())

    def test_repeated_hashtag_added_only_once(self):
        self.caption.save_user_caption("#sun and more #sun", commit=True)

        album = AlbumThing.objects.get(
            title="#sun", thing_type="hashtag_attribute", owner=self.user
        )
        self.assertEqual(album.photos.count(), 1)

    def test_hashtag_removed_when_no_longer_in_caption(self):
        self.caption.save_user_caption("#sun #sea", commit=True)
        self.caption.save_user_caption("#sun only", commit=True)

        sun = AlbumThing.objects.get(
            title="#sun", thing_type="hashtag_attribute", owner=self.user
        )
        sea = AlbumThing.objects.get(
            title="#sea", thing_type="hashtag_attribute", owner=self.user
        )
        # The AlbumThing row is kept; only the photo association is removed.
        self.assertEqual(sun.photos.count(), 1)
        self.assertEqual(sea.photos.count(), 0)

    def test_removal_uses_substring_match_not_token_match(self):
        """QUIRK: '#sun' survives because it is a substring of '#sunset'."""
        self.caption.save_user_caption("#sun", commit=True)
        self.caption.save_user_caption("#sunset", commit=True)

        sun = AlbumThing.objects.get(
            title="#sun", thing_type="hashtag_attribute", owner=self.user
        )
        self.assertEqual(sun.photos.count(), 1)

    def test_hashtag_albums_of_other_owner_untouched(self):
        other = create_test_user()
        other_album = get_album_thing(
            title="#sun", owner=other, thing_type="hashtag_attribute"
        )
        other_album.photos.add(self.photo)

        self.caption.save_user_caption("nothing tagged", commit=True)

        other_album.refresh_from_db()
        self.assertEqual(other_album.photos.count(), 1)

    def test_non_hashtag_album_types_untouched(self):
        places_album = get_album_thing(
            title="beach", owner=self.user, thing_type="places365_attribute"
        )
        places_album.photos.add(self.photo)

        self.caption.save_user_caption("no tags here", commit=True)

        places_album.refresh_from_db()
        self.assertEqual(places_album.photos.count(), 1)

    # ---------- error branch ----------

    def test_returns_false_when_caption_is_not_a_string(self):
        """AttributeError inside the try block is swallowed -> False."""
        result = self.caption.save_user_caption(None, commit=True)

        self.assertFalse(result)
        self.assertIsNone(self.caption.captions_json)

    def test_returns_false_when_inner_call_raises(self):
        with patch.object(
            PhotoCaption, "recreate_search_captions", side_effect=RuntimeError("boom")
        ):
            result = self.caption.save_user_caption("hello", commit=True)

        self.assertFalse(result)
        # The mutation before the raise is still present on the instance.
        self.assertEqual(self.caption.captions_json["user_caption"], "hello")
        fresh = PhotoCaption.objects.get(photo=self.photo)
        self.assertIsNone(fresh.captions_json)
