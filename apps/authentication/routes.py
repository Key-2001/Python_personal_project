# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template, redirect, request, url_for, Response
import cv2
import numpy as np
import os
import sqlite3
from PIL import Image
from flask_login import (
    current_user,
    login_user,
    logout_user
)
import pandas as pd
from datetime import datetime
from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users
from apps.authentication.util import verify_pass

df = pd.read_csv('advanced_python.csv',sep=';')
df2 =pd.read_csv('winemag-data-130k-v2.csv')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/macbookpro/Workspace/Advance_python/week3/firstapplication/faceRecognition/recognizer/trainingData.yml')

def getProfile(id):
    con = sqlite3.connect('/Users/macbookpro/Workspace/Advance_python/week3/firstapplication/faceRecognition/data.db')
    cur = con.cursor()

    query = "SELECT * FROM student WHERE id='"+str(id)+"'"
    cursorr = cur.execute(query)

    profile = None

    for row in cursorr:
        profile = row
    
    con.close()
    return profile

def gen_frames():
    cap = cv2.VideoCapture(0)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray)
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]

        id, confidence = recognizer.predict(roi_gray)
        id = 'A' + str(id)
        # print('id',id)
        if confidence< 40:
            profile = getProfile(id)
            print(profile)
            if(profile != None):
                cv2.putText(frame,""+str(profile[0])+"_"+str(profile[1]),(x+10,y+h+30),fontface,1,(0,255,0),2)
            
        else:
            cv2.putText(frame,"Unknown",(x+10,y+h+30),fontface,1,(0,0,255),2)
        cv2.imshow('SHOW',frame)
        if(cv2.waitKey(1) == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

@blueprint.route('/')
def route_default():
    return redirect(url_for('authentication_blueprint.login'))


# Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:

        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = Users.query.filter_by(username=username).first()

        # Check the password
        if user and verify_pass(password, user.password):

            login_user(user)
            return redirect(url_for('authentication_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template('accounts/login.html',
                               msg='Wrong user or password',
                               form=login_form)

    if not current_user.is_authenticated:
        return render_template('accounts/login.html',
                               form=login_form)
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']
        role = request.form['role']
        # photo = request.form['photo']
        

        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Username already registered',
                                   success=False,
                                   form=create_account_form)

        # Check email exists
        user = Users.query.filter_by(email=email).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Email already registered',
                                   success=False,
                                   form=create_account_form)

        # else we can create the user
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()

        return render_template('accounts/register.html',
                               msg='User created please <a href="/login">login</a>',
                               success=True,
                               form=create_account_form)

    else:
        return render_template('accounts/register.html', form=create_account_form)


@blueprint.route('/profile',methods=['GET'])
def profile():

    return render_template('accounts/profile.html',user=current_user)

@blueprint.route('/student',methods=['GET','POST'])
def students():
    if(request.method == 'POST'):
        result = request.form['studentCode'];
        student = df[df['student code'] == result];
        studentCode = student['student code']
        firstName = student['First name']
        lastName = student['Last name']
        DOB = student['DOB']
        CN = student['CN']
        length = 0;
        no = pd.to_numeric(student['No'].to_string()[0:2]);
        return render_template('accounts/students.html',no=no, studentCode=studentCode,firstName=firstName,lastName=lastName,DOB=DOB,CN=CN,length=length)
    studentCode = df['student code']
    firstName = df['First name']
    lastName = df['Last name']
    DOB = df['DOB']
    CN = df['CN']
    length = len(df)
    return render_template('accounts/students.html', studentCode=studentCode,firstName=firstName,lastName=lastName,DOB=DOB,CN=CN,length=length)

@blueprint.route('/student/<string:code>',methods=['GET'])
def student(code):
    student = df[df['student code'] == code];
    # no = pd.to_numeric(student['No'].to_string()[0]);
    no = pd.to_numeric(student['No'].to_string()[0:2]);
    studentCode = student['student code'];
    firstName = student['First name'];
    lastName = student['Last name'];
    DOB = student['DOB'];
    CN = student['CN'];
    query = ''
    return render_template('accounts/singleStudent.html',no=no,studentCode=studentCode,firstName=firstName,lastName=lastName,DOB=DOB,CN=CN,query=query);

@blueprint.route('/wine',methods=['GET'])
def wine():
    
    country = df2.groupby(['country'])['Unnamed: 0'].count()[df2.groupby(['country']).count().description > 2000]
    country = country.to_string().split();
    length = len(country)
    arrCountry=[];
    arrAmount=[]
    for i in range(1,length):
        if(i %2 != 0):
            arrCountry.append(country[i]);
        else:
            arrAmount.append(pd.to_numeric(country[i]));
    test = df2['country'];
    country2 = df2.groupby(['country']).sum()[df2.groupby(['country']).count().description > 2000]
    arrPoint = country2['points'];
    arrPrice = country2['price'];
    length_ = len(arrPoint)
    medPoint = [];
    medPrice = [];
    for i in range(0,length_):
        medPoint.append(round(arrPoint[i]/arrAmount[i],2))
        medPrice.append(round(arrPrice[i]/arrAmount[i],2))

    lengthArr = len(arrCountry)
    return render_template('accounts/chart.html',arrCountry=arrCountry,arrAmount=arrAmount,test = test,length = lengthArr,
                            medPoint=medPoint,medPrice=medPrice,arrPoint=arrPoint,arrPrice=arrPrice,country2=country2)
@blueprint.route('/face')
def face():
    return render_template('accounts/face.html')

@blueprint.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login'))


    
# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
