from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import User

# Create your models here.
class PVString(models.Model):
    Username = models.ForeignKey(User, db_index=True, on_delete=models.CASCADE,)
    PVCode = models.CharField(_('PVCode'), max_length=50)
    Azimuth = models.FloatField(_('Azimuth'), max_length=50)
    Tilt = models.FloatField(_('Tilt'))
    Pnom = models.FloatField(_('Pnominal'), max_length=250)
    slug = models.SlugField(max_length=200,	unique=True)

    class Meta:
        ordering = ('Username',)
        verbose_name = 'PVString'
        verbose_name_plural = 'PVStrings'
    
    def __str__(self): 
        return 'PVCode={0}, Azimuth={1}, Pnominal={2}'.format(self.PVCode, self.Azimuth, self.Pnom)
        
    def get_absolute_url(self):
        return reverse('shop:pvstring_by_username',args=[self.slug])

irr_range = range(1,10,1)
irr_range =  [ (i, str(i)) for i in irr_range ]
     
class DayIrradiance(models.Model):
    DateNow = models.DateField(_('DateNow'))
    IrrClass = models.IntegerField(choices=irr_range, default=5)
    slug = models.SlugField(max_length=200,	unique=True)
    
    class Meta:
        ordering = ('DateNow',)
        verbose_name = 'DayIrradiance'
        verbose_name_plural = 'DayIrradiances'
    
    def __str__(self): 
        return 'Date={0}, IrrClass={1}'.format(self.DateNow, self.IrrClass)
        
    def get_absolute_url(self):
        return reverse('shop:dayirr_by_date',args=[self.slug])