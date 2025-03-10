import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch credentials
sender_email = "aiotcraftvigilante@gmail.com"
receiver_email = "ujjawalgarg7@gmail.com"
password = os.getenv("MAIL_API")  # Load password from .env

# Ensure the password is loaded correctly
if not password:
    raise ValueError("MAIL_API is not set. Check your .env file or environment variables.")

# Create email content
msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "SOS MAIL"

body = "PLEASE HELP ME"
msg.attach(MIMEText(body, "plain"))

# Sending the email via Gmail SMTP
try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Secure the connection
    server.login(sender_email, password)  # Login with the correct password
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
    print("Email sent successfully!")
except smtplib.SMTPAuthenticationError:
    print("Authentication failed. Check if the password (MAIL_API) is correct or if 'Less Secure Apps' is enabled.")
except Exception as e:
    print(f"Error: {e}")
