"""Person edits keep the names indexed by face labelling in sync."""

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from rest_framework.test import APIClient

from api.models.photo_search import PhotoSearch
from api.tests.utils import (
    create_test_face,
    create_test_person,
    create_test_photo,
    create_test_user,
)


class PersonSearchCaptionsTest(TestCase):
    def setUp(self) -> None:
        self.user = create_test_user()
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.person = create_test_person(name="Alice", cluster_owner=self.user)
        self.url = f"/api/persons/{self.person.pk}/"
        self.photos = [
            create_test_photo(
                owner=self.user,
                captions_json={"user_caption": "Family portrait"},
                camera_model="X-T5",
                search_location="Zürich",
                **fields,
            )
            for fields in ({}, {"hidden": True}, {"in_trashcan": True})
        ]
        faces = [create_test_face(photo=photo) for photo in self.photos]
        # Several faces of the person on one photo must not duplicate its refresh.
        faces.append(create_test_face(photo=self.photos[0]))
        other = create_test_person(name="Bob", cluster_owner=self.user)
        create_test_face(photo=self.photos[0], person=other)
        response = self.client.post(
            "/api/labelfaces",
            {"person_name": self.person.name, "face_ids": [face.pk for face in faces]},
            format="json",
        )
        self.assertEqual(response.status_code, 200)

    def test_rename_refreshes_every_affected_photo(self) -> None:
        response = self.client.patch(
            self.url, {"newPersonName": "Élodie"}, format="json"
        )

        self.assertEqual(response.status_code, 200)
        self.person.refresh_from_db()
        self.assertEqual(self.person.name, "Élodie")
        for photo in self.photos:
            with self.subTest(photo=photo.pk):
                search = PhotoSearch.objects.get(photo=photo)
                self.assertIn("Élodie", search.search_captions)
                self.assertNotIn("Alice", search.search_captions)
                self.assertIn("elodie", search.search_folded)
                self.assertNotIn("alice", search.search_folded)
                self.assertIn("Family portrait", search.search_captions)
                self.assertIn("X-T5", search.search_captions)
                self.assertEqual(search.search_location, "Zürich")
                self.assertIn("zurich", search.search_folded)
        self.assertIn(
            "Bob", PhotoSearch.objects.get(photo=self.photos[0]).search_captions
        )

    def test_delete_refreshes_every_affected_photo(self) -> None:
        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 204)
        for photo in self.photos:
            with self.subTest(photo=photo.pk):
                search = PhotoSearch.objects.get(photo=photo)
                self.assertNotIn("Alice", search.search_captions)
                self.assertNotIn("alice", search.search_folded)
                self.assertIn("Family portrait", search.search_captions)
                self.assertIn("X-T5", search.search_captions)
                self.assertEqual(search.search_location, "Zürich")
                self.assertIn("zurich", search.search_folded)
        self.assertIn(
            "Bob", PhotoSearch.objects.get(photo=self.photos[0]).search_captions
        )

    def test_edits_create_missing_search_rows(self) -> None:
        photo = self.photos[-1]
        PhotoSearch.objects.filter(photo=photo).delete()

        response = self.client.patch(
            self.url, {"newPersonName": "Élodie"}, format="json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Élodie", PhotoSearch.objects.get(photo=photo).search_captions)
        PhotoSearch.objects.filter(photo=photo).delete()

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 204)
        self.assertNotIn("Élodie", PhotoSearch.objects.get(photo=photo).search_captions)

    def test_edits_preserve_other_sources_of_the_old_name(self) -> None:
        photo = self.photos[0]
        other = create_test_person(name="Alice Cooper", cluster_owner=self.user)
        create_test_face(photo=photo, person=other)
        caption = photo.caption_instance
        caption.captions_json = {"user_caption": "Alice in Wonderland"}
        caption.save()

        response = self.client.patch(
            self.url, {"newPersonName": "Élodie"}, format="json"
        )

        self.assertEqual(response.status_code, 200)
        search = PhotoSearch.objects.get(photo=photo)
        self.assertIn("Élodie", search.search_captions)
        self.assertIn("Alice in Wonderland", search.search_captions)
        self.assertIn("Alice Cooper", search.search_captions)

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 204)
        search.refresh_from_db()
        self.assertNotIn("Élodie", search.search_captions)
        self.assertIn("Alice in Wonderland", search.search_captions)
        self.assertIn("Alice Cooper", search.search_captions)

    def test_edits_do_not_refresh_unrelated_or_other_users_photos(self) -> None:
        unrelated = create_test_photo(owner=self.user, search_captions="Alice")
        theirs = create_test_photo(owner=create_test_user(), search_captions="Alice")
        # A legacy cross-owner face link must not make this a write to their index.
        create_test_face(photo=theirs, person=self.person)
        searches = PhotoSearch.objects.filter(photo__in=[unrelated, theirs]).order_by(
            "photo_id"
        )
        fields = ("search_captions", "search_folded", "updated_at")
        before = list(searches.values_list(*fields))

        response = self.client.patch(
            self.url, {"newPersonName": "Élodie"}, format="json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(before, list(searches.values_list(*fields)))

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 204)
        self.assertEqual(before, list(searches.values_list(*fields)))

    def test_changing_the_cover_does_not_rebuild_search(self) -> None:
        search = PhotoSearch.objects.get(photo=self.photos[0])
        updated_at = search.updated_at

        response = self.client.patch(
            self.url, {"cover_photo": str(self.photos[0].pk)}, format="json"
        )

        self.assertEqual(response.status_code, 200)
        search.refresh_from_db()
        self.assertEqual(search.updated_at, updated_at)

    def test_person_without_faces_can_be_renamed_and_deleted(self) -> None:
        person = create_test_person(name="Carol", cluster_owner=self.user)
        url = f"/api/persons/{person.pk}/"

        response = self.client.patch(url, {"newPersonName": "Caroline"}, format="json")

        self.assertEqual(response.status_code, 200)
        person.refresh_from_db()
        self.assertEqual(person.name, "Caroline")
        self.assertEqual(self.client.delete(url).status_code, 204)

    def test_edits_rebuild_search_with_a_bounded_query_count(self) -> None:
        query_counts = []
        for count in (2, 8):
            person = create_test_person(cluster_owner=self.user)
            url = f"/api/persons/{person.pk}/"
            for _ in range(count):
                photo = create_test_photo(owner=self.user, search_captions=person.name)
                create_test_face(photo=photo, person=person)

            with CaptureQueriesContext(connection) as rename_queries:
                response = self.client.patch(
                    url, {"newPersonName": "Renamed"}, format="json"
                )
            self.assertEqual(response.status_code, 200)
            with CaptureQueriesContext(connection) as delete_queries:
                response = self.client.delete(url)
            self.assertEqual(response.status_code, 204)
            query_counts.append((len(rename_queries), len(delete_queries)))

        self.assertEqual(query_counts[0], query_counts[1])
