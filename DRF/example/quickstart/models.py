from django.db import models


class Person(models.Model):
    firstname = models.CharField("First name", max_length=50)
    lastname = models.CharField("Last name", max_length=50)
    age = models.PositiveIntegerField("Age")

    class Meta:
        ordering = ['lastname']
