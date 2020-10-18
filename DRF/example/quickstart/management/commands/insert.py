from django.core.management.base import BaseCommand
from django.utils import timezone
from quickstart.models import Person

class Command(BaseCommand):
    help = 'Displays current time'

    def handle(self, *args, **kwargs):

        for i in range(20):
            person = Person(
                firstname="Adam" + str(i),
                lastname="Kowalski" + str(i),
                age=i
            )
            person.save()
