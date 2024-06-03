import requests
import os


def notify_telegram_group(message):
    bot_token = os.getenv("TRAINER_BOT_TOKEN", "")
    chat_id = os.getenv("TRAINER_BOT_CHAT_ID", "")
    if bot_token == "":
        return ""

    if chat_id == "":
        return ""

    return telegram_send_message(bot_token, chat_id, message)


def send_photo_telegram_group(image_path, caption):
    bot_token = os.getenv("TRAINER_BOT_TOKEN", "")
    chat_id = os.getenv("TRAINER_BOT_CHAT_ID", "")
    if bot_token == "":
        return ""

    if chat_id == "":
        return ""

    return send_photo(bot_token, chat_id, image_path, caption)


def telegram_send_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, json=payload)
    return response.json()


def send_photo(bot_token, chat_id, image_path, caption):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {"photo": open(image_path, "rb")}
    data = {"chat_id": chat_id, "caption": caption}
    response = requests.post(url, files=files, data=data)
    return response.json()
