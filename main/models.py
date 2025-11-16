from django.db import models

# Create your models here.


class question(models.Model):
    user_address = models.CharField(max_length = 150)
    user_question = models.TextField()
    model_answer = models.TextField()

    def __str__(self):
        return self.user_address