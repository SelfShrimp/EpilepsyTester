from django.db import models

# Create your models here.

from django import forms


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    unique_name = models.CharField(max_length=255, unique=True)