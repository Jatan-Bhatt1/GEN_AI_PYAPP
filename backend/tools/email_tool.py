"""
Email Tool — Send emails via SMTP (Gmail, Outlook, or any SMTP server).
Uses Python's built-in smtplib with TLS encryption.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain_core.tools import tool
from pydantic import BaseModel, Field, EmailStr
from loguru import logger
from backend.config import get_settings

settings = get_settings()


class SendEmailInput(BaseModel):
    to_email: str = Field(description="Recipient email address, e.g. 'user@example.com'")
    subject: str = Field(description="Email subject line")
    body: str = Field(
        description=(
            "Email body content. Use plain text or HTML. "
            "For HTML, wrap content in <html><body>...</body></html> tags."
        )
    )
    is_html: bool = Field(
        default=False,
        description="Set to True if the body contains HTML markup",
    )


@tool("send_email", args_schema=SendEmailInput)
def send_email_tool(to_email: str, subject: str, body: str, is_html: bool = False) -> str:
    """
    Send an email to a specified recipient using the configured SMTP server.
    Use this when the user explicitly asks you to send, draft, or email something.
    Always confirm with the user before sending if the request is ambiguous.

    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body (plain text or HTML)
        is_html: Whether body contains HTML (default: False)

    Returns:
        Success confirmation with message ID, or error description
    """
    # Read SMTP config from settings
    smtp_host = getattr(settings, "smtp_host", "smtp.gmail.com")
    smtp_port = getattr(settings, "smtp_port", 587)
    smtp_user = getattr(settings, "smtp_user", "")
    smtp_password = getattr(settings, "smtp_password", "")
    smtp_from = getattr(settings, "smtp_from", smtp_user)

    if not smtp_user or not smtp_password:
        return (
            "Email tool is not configured. "
            "Please set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and SMTP_FROM "
            "in your .env file."
        )

    try:
        # Build the MIME message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email

        content_type = "html" if is_html else "plain"
        msg.attach(MIMEText(body, content_type, "utf-8"))

        logger.info(f"Sending email to {to_email}: '{subject}'")

        # Connect and send via STARTTLS
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, to_email, msg.as_string())

        logger.success(f"Email sent to {to_email}")
        return (
            f"✅ Email sent successfully!\n"
            f"  To: {to_email}\n"
            f"  Subject: {subject}\n"
            f"  From: {smtp_from}"
        )

    except smtplib.SMTPAuthenticationError:
        return (
            "SMTP authentication failed. "
            "For Gmail, use an App Password (not your regular password). "
            "Enable 2FA and create an App Password at: myaccount.google.com/apppasswords"
        )
    except smtplib.SMTPRecipientsRefused:
        return f"Email delivery failed: recipient '{to_email}' was rejected by the server."
    except TimeoutError:
        return f"SMTP connection timed out connecting to {smtp_host}:{smtp_port}."
    except Exception as e:
        logger.error(f"Email sending error: {e}")
        return f"Failed to send email: {str(e)}"
