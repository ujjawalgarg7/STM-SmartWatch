import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import requests
import os
from location import get_location
# Load environment variables from .env file
load_dotenv()

# Fetch credentials
sender_email = "aiotcraftvigilante@gmail.com"
password = os.getenv("MAIL_API")  # Load password from .env


# Function to send an emergency email
def sendMail(receiver_email, message):
    # Ensure the password is loaded correctly
    if not password:
        raise ValueError("MAIL_API is not set. Check your .env file or environment variables.")

    # Get location
    location_data = get_location()

    if "error" in location_data:
        location_text = "Location not available."
    else:
        location_text = f"""
        📍 Location: {location_data["city"]}, {location_data["country"]}
        🌍 Latitude, Longitude: {location_data["latitude"]}, {location_data["longitude"]}
        📌 Google Maps Link: {location_data["map_link"]}
        """

    # Create email content
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "SOS EMERGENCY"

    body = "Emergency!!! Your friend needs help!"

    if message == "FALLEN":
        body = f"""URGENT: A person has fallen and needs immediate help!

        {location_text}

        the person has fallen while walking/running.

        Immediate assistance required! Time is critical! ⏳"""
    
    elif message == "GESTURE":
        body = f"""⚠️ LIFE-THREATENING SITUATION! ⚠️

        A person is in danger and signaling for help! Immediate action is required!

        {location_text}

        
        Please respond immediately! This is a CRITICAL emergency! 🚔🚑⏳"""

    msg.attach(MIMEText(body, "plain"))

    # Sending the email via Gmail SMTP
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, password)  # Login with the correct password
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return "Email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return "Authentication failed. Check if the password (MAIL_API) is correct or if 'Less Secure Apps' is enabled."
    except Exception as e:
        return f"Error: {e}"

# Example Usage
result = sendMail("receiver@example.com", "FALLEN")
print(result)  # Can be removed if not needed
