from django import forms
import datetime
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column

# now time
now = datetime.datetime.now()

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
    
class ShowPVgen(forms.Form):
    user_fullname = forms.CharField(label="Username",max_length=100,widget=forms.TextInput(attrs={'size':500,'readonly':'readonly'}))
    user_pv_code = forms.CharField(label="Κωδικός ΦΒ συστήματος",max_length=100,widget=forms.TextInput(attrs={'size':500,'readonly':'readonly'}))
    date_now = forms.DateTimeField(required=False, label="Ημερομηνία", widget=forms.TextInput(attrs={'readonly':'readonly'}))
    pv_gen = forms.FloatField(required=False,label="Ημερήσια παραγωγή ηλεκτρικής ενέργειας (kWh)",widget=forms.TextInput(attrs={'readonly':'readonly'}))
    
class ShowPVgenF(forms.Form):
    user_fullname = forms.CharField(label="Ονοματεπώνυμο",max_length=100,widget=forms.TextInput(attrs={'readonly':'readonly'}))
    user_pv_code = forms.CharField(label="Κωδικός ΦΒ συστήματος",max_length=100,widget=forms.TextInput(attrs={'readonly':'readonly'}))
    date_now = forms.DateTimeField(required=False, label="Ημερομηνία", widget=forms.TextInput(attrs={'readonly':'readonly'}))
    pv_gen = forms.FloatField(required=False,label="Παραγωγή ηλεκτρικής ενέργειας (kWh/ημέρα)",widget=forms.TextInput(attrs={'readonly':'readonly'}))
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Row(
                Column('user_fullname', css_class='form-group col-md-4 mb-0'),
                Column('user_pv_code', css_class='form-group col-md-4 mb-0'),
                css_class='form-row'
            ),
            
            Row(
                Column('date_now', css_class='form-group col-md-4 mb-0'),
                Column('pv_gen', css_class='form-group col-md-4 mb-0'),
                css_class='form-row'
            ),
            
            Submit('submit', 'Ανανέωση αποτελέσματος')
        )