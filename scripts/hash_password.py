"""Generate a password hash for the webapp login.

Usage:
    python scripts/hash_password.py
    (you'll be prompted; the hash goes into APP_PASSWORD_HASH in your .env)
"""
from getpass import getpass

from werkzeug.security import generate_password_hash


def main() -> None:
    pw = getpass("Password: ")
    pw2 = getpass("Confirm:  ")
    if pw != pw2:
        print("Mismatch.")
        return
    print()
    print("Set this in your .env file:")
    print(f"APP_PASSWORD_HASH={generate_password_hash(pw)}")


if __name__ == "__main__":
    main()
