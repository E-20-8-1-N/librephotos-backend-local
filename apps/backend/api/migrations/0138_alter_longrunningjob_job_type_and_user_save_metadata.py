from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0137_photo_ocr_source_dimensions"),
    ]

    operations = [
        migrations.AlterField(
            model_name="longrunningjob",
            name="job_type",
            field=models.PositiveIntegerField(
                choices=[
                    (1, "Scan Photos"),
                    (2, "Generate Event Albums"),
                    (3, "Regenerate Event Titles"),
                    (4, "Train Faces"),
                    (5, "Delete Missing Photos"),
                    (7, "Scan Faces"),
                    (6, "Calculate Clip Embeddings"),
                    (8, "Find Similar Faces"),
                    (9, "Download Selected Photos"),
                    (10, "Download Models"),
                    (11, "Add Geolocation"),
                    (12, "Generate Tags"),
                    (13, "Generate Face Embeddings"),
                    (14, "Scan Missing Photos"),
                    (15, "Detect Duplicate Photos"),
                    (16, "Repair File Variants"),
                    (17, "Classify Media Categories"),
                    (18, "Extract Text (OCR)"),
                    (19, "Generate im2txt Captions"),
                ]
            ),
        ),
        migrations.AlterField(
            model_name="user",
            name="save_metadata_to_disk",
            field=models.TextField(
                choices=[
                    ("OFF", "Off"),
                    ("MEDIA_FILE", "Media File"),
                    ("SIDECAR_FILE", "Sidecar File"),
                ],
                default="MEDIA_FILE",
            ),
        ),
    ]
