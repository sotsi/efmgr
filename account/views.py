from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth import authenticate, login
from .forms import LoginForm, ShowPVgen, ShowPVgenF
from .models import PVString, DayIrradiance
from django.contrib.auth.decorators import login_required
import random 
import datetime
import os
import pickle
import numpy as np

cwdp = os.getcwd()

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(request, username=cd['username'], password=cd['password'])
        if user is not None:
            if user.is_active:
                login(request, user)
                return HttpResponse('Authenticated successfully')
            else:
                return HttpResponse('Disabled account')
        else:
            return HttpResponse('Invalid login')
    else:
        form = LoginForm()
    return render(request, 'account/login.html', {'form': form})
    
@login_required
def dashboard(request):
    current_user = request.user.username
    current_id = request.user.id
    # get pv strings
    pvstringsl = PVString.objects.filter( Username=current_id ).count()
    pvstrings = PVString.objects.filter( Username=current_id )
    if pvstringsl>0:
        pvcode = pvstrings[0].PVCode
    else:
        pvcode = 'None'
    # get day irradiance class
    dayirr = DayIrradiance.objects.all().last().IrrClass
    daylast = DayIrradiance.objects.all().last().DateNow
    # return HttpResponse(dayirr)
    ### load and run svm
    now = datetime.datetime.now()
    cur_month = now.month
    monthcos = np.cos(2 * np.pi * cur_month/12.0)
    strl = [cwdp, '\\account\static\svm\\', current_user, '.pickle']
    svm_path =  ''.join(strl)
    # load the model from disk
    try:
        loaded_model = pickle.load(open(svm_path, 'rb'))
        result = loaded_model.predict(np.array([dayirr, monthcos]).reshape(1,-1))[0]
    except:
        result = 0
    # return HttpResponse(result)
    ############
    
    if request.method == 'POST':
        form = ShowPVgen(request.POST)
        if form.is_valid():
            user_fullname = request.POST.get('user_fullname')
            user_pv_code = request.POST.get('user_pv_code')
            date_now = request.POST.get('date_now')
            pv_gen = result # round(random.uniform(0.0, 10.0), 1)
            data = {'user_fullname': current_user, 'user_pv_code': pvcode, 'date_now': daylast, 'pv_gen': result}
            form = ShowPVgen(data)
            return render(request, 'account/dashboard.html', {'pvs':pvstrings, 'form': form, 'msg': 'ok button pressed...', 'section': 'dashboard'})
    else:
        data = {'user_fullname': current_user, 'user_pv_code': pvcode, 'date_now': daylast, 'pv_gen': result}
        form = ShowPVgen(data)
        return render(request, 'account/dashboard.html', {'pvs':pvstrings, 'form': form, 'msg': 'executed...', 'section': 'dashboard'})

    return render(request, 'account/dashboard.html', {'form': form, 'msg': 'ok...', 'section': 'dashboard'})