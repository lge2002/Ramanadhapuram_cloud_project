from django.db import models

class Cloud_ramanathapuram(models.Model):
    city = models.CharField(max_length=100) 
    values = models.TextField() 
    type = models.CharField(max_length=50, default="Weather radar")
    timestamp = models.DateTimeField(db_index=True) 

    class Meta:
        unique_together = ('city', 'timestamp',)

    def __str__(self):
        return f"{self.city} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.values[:50]}..."