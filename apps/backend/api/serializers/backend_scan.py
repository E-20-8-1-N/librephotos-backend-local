"""The path-based integration API's request and response."""

from rest_framework import serializers


class BackendPathRequest(serializers.Serializer[None]):
    path = serializers.CharField()


class BackendJobResponse(serializers.Serializer[None]):
    status = serializers.BooleanField()
    job_id = serializers.UUIDField()
    username = serializers.CharField()
    path = serializers.CharField()
    scan_directory = serializers.CharField(required=False)
