# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 19:25:21 2018
@author: ME389019
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import  MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
subject="HI PYTHON SCRIPTS"
#mail.ehlo()
usermail='iamatester1995@gmail.com'
receiver='medha.rwt@gmail.com'

msg=MIMEMultipart()
msg['From']=usermail
msg['To']=receiver
msg['Subject']=subject
body = "HEY via py script I am able to send u mail"
msg.attach(MIMEText(body,'plain'))

#filename='E:\\EA\\studies\\doc.txt'
filename="doc.txt"
#filename="me.jpg"
#filename='E:\\EA\\studies\\me.jpg'
attachment=open("E:\\EA\\studies\\doc.txt","rb")

#alllowing the uploading and streaming of the attachment
part = MIMEBase('application','octet-stream')
part.set_payload((attachment).read())

encoders.encode_base64(part) # we will encode the attachments(check bases - base64)
part.add_header('Content-Disposition',"attachment;filename=%s"%filename)

msg.attach(part)
text=msg.as_string()

mail =smtplib.SMTP('smtp.gmail.com',587)
mail.starttls()
mail.login(usermail,'testing22091995')
mail.sendmail(usermail,receiver,text)
mail.quit()

print(attachment.read())