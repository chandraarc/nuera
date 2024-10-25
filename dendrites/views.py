from django.shortcuts import render
from django.http import HttpResponse
from dendrites_service import handle_uploaded_file

def members(request):
    handle_uploaded_file(request.FILES['file'])
    return HttpResponse("file uploaded")