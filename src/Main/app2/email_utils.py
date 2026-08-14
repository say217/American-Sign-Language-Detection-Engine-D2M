import os
import secrets
import smtplib
from email.message import EmailMessage
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = get_env_int("SMTP_PORT", 587)
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER = os.getenv("SMTP_SENDER", "") or SMTP_USER
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
VERIFY_TOKEN_TTL_MINUTES = get_env_int("VERIFY_CODE_EXP_MINUTES", 10)

ICON_PATH = Path(os.getenv("EMAIL_ICON_PATH", Path(__file__).parent / "ai_icon.png"))
ICON_CID = "ai_icon"

def _build_plain_text(code: str) -> str:
    return (
        "Verify your account\n\n"
        "Hi there,\n\n"
        "Thanks for signing up! To finish creating your account and make sure "
        "this email address belongs to you, please enter the verification code "
        "below in the app or website where you started signing up.\n\n"
        f"Your verification code: {code}\n\n"
        f"This code will expire in {VERIFY_TOKEN_TTL_MINUTES} minutes, so be "
        "sure to use it soon.\n\n"
        "For your security, never share this code with anyone, including "
        "someone claiming to be from our support team - we will never ask "
        "you for it.\n\n"
        "If you did not attempt to sign up or request this code, you can "
        "safely ignore this email. No account will be created and no action "
        "is needed on your part.\n\n"
        "Thanks,\n"
        "The Team"
    )


def _build_html(code: str) -> str:
    icon_html = (
        f'<img src="cid:{ICON_CID}" width="56" height="56" alt="AI" '
        'style="display:block;border-radius:14px;margin:0 auto 20px auto;" />'
        if ICON_PATH.exists()
        else ""
    )

    return f"""\
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Welcome to Sign Bridge AI Verify your account</title>
  </head>
  <body style="margin:0;padding:0;background-color:#0b0b0d;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#0b0b0d;padding:40px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="480" cellpadding="0" cellspacing="0"
                 style="background-color:#161618;border:1px solid #2a2a2e;border-radius:16px;padding:40px 32px;">
            <tr>
              <td align="center">
                {icon_html}
                <h1 style="margin:0 0 8px 0;font-size:22px;line-height:28px;color:#f5f5f7;font-weight:600;">
                  Verify your account
                </h1>
                <p style="margin:0 0 28px 0;font-size:14px;line-height:22px;color:#9a9aa2;">
                  Enter the code below to confirm it's really you.
                </p>

                <div style="background-color:#0b0b0d;border:1px solid #2a2a2e;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
                  <span style="font-size:34px;font-weight:700;letter-spacing:10px;color:#ffffff;">
                    {code}
                  </span>
                </div>

                <p style="margin:0 0 24px 0;font-size:13px;line-height:20px;color:#7a7a82;">
                  This code expires in <strong style="color:#c7c7cf;">{VERIFY_TOKEN_TTL_MINUTES} minutes</strong>.
                  For your security, never share it with anyone - we'll never ask you for it directly.
                </p>

                <hr style="border:none;border-top:1px solid #2a2a2e;margin:24px 0;" />

                <p style="margin:0;font-size:12px;line-height:19px;color:#5c5c63;">
                  If you didn't request this code, you can safely ignore this email.
                  No changes will be made to your account.
                </p>
              </td>
            </tr>
          </table>

          <p style="margin:24px 0 0 0;font-size:11px;color:#4b4b52;">
            Sent from {BASE_URL}
          </p>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def send_verification_email(recipient_email: str, code: str) -> str | None:
    """Sends a dark-themed HTML verification email (with plain-text fallback).

    Returns None on success, or an error message string on failure.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        return "SMTP credentials are missing. Set SMTP_USER and SMTP_PASSWORD."

    message = EmailMessage()
    message["Subject"] = "Verify your account"
    message["From"] = SMTP_SENDER
    message["To"] = recipient_email

    # Plain-text fallback for clients that don't render HTML.
    message.set_content(_build_plain_text(code))

    # HTML (dark theme) alternative.
    message.add_alternative(_build_html(code), subtype="html")

    # Attach the inline "AI" icon so it can be referenced via cid: in the HTML.
    if ICON_PATH.exists():
        html_part = message.get_payload()[-1]
        with open(ICON_PATH, "rb") as f:
            html_part.add_related(
                f.read(),
                maintype="image",
                subtype="png",
                cid=f"<{ICON_CID}>",
                filename=ICON_PATH.name,
            )

    try:
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.ehlo()
                if SMTP_USE_TLS:
                    server.starttls()
                    server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
    except smtplib.SMTPAuthenticationError:
        return (
            "SMTP authentication failed. If you're using Gmail, make sure you're using a "
            "16-character App Password (not your normal Gmail password) and that "
            "2-Step Verification is enabled on the account."
        )
    except smtplib.SMTPException as exc:
        return f"Failed to send email: {exc}"
    except OSError as exc:
        return f"Could not connect to SMTP server: {exc}"

    return None


def generate_code() -> str:
    return f"{secrets.randbelow(1000000):06d}"