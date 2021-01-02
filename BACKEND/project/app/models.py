from django.db import models


class Progress(models.Model):
    name = models.CharField("Name", max_length = 50)
    progress = models.PositiveIntegerField("Progress")

    class Meta:
        ordering = ['id']