import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import  MIMEMultipart
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
text=msg.as_string()

mail =smtplib.SMTP('smtp.gmail.com',587)
mail.starttls()
mail.login(usermail,'testing22091995')
mail.sendmail(usermail,receiver,text)
mail.quit()