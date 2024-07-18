import requests
import os


def notify_telegram_group(message) -> int:
    bot_token = os.getenv("TRAINER_BOT_TOKEN", "")
    chat_id = os.getenv("TRAINER_BOT_CHAT_ID", "")
    if bot_token == "":
        return ""

    if chat_id == "":
        return ""

    return telegram_send_message(bot_token, chat_id, message)


def update_notify_telegram_group(message_id, message) -> int:
    bot_token = os.getenv("TRAINER_BOT_TOKEN", "")
    chat_id = os.getenv("TRAINER_BOT_CHAT_ID", "")
    if bot_token == "":
        return ""

    if chat_id == "":
        return ""
    if message_id == None or message_id == 0:
        try:
            return telegram_send_message(bot_token, chat_id, message)
        except Exception as e:
            print(f"An error occurred: {e}")
            return message_id
    try:
        return telegram_update_message(bot_token, chat_id, message_id, message)
    except Exception as e:
        print(f"An error occurred: {e}")
        return message_id


def send_photo_telegram_group(image_path, caption):
    bot_token = os.getenv("TRAINER_BOT_TOKEN", "")
    chat_id = os.getenv("TRAINER_BOT_CHAT_ID", "")
    if bot_token == "":
        return ""

    if chat_id == "":
        return ""

    return send_photo(bot_token, chat_id, image_path, caption)


def telegram_send_message(bot_token, chat_id, message) -> int:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, json=payload)
    response_data = response.json()

    if response_data["ok"]:
        return response_data["result"]["message_id"]
    else:
        raise Exception(f"Error sending message: {response_data}")


def telegram_update_message(
    bot_token: str, chat_id: str, message_id: int, new_text: str
):
    url = f"https://api.telegram.org/bot{bot_token}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": new_text,
        "parse_mode": "Markdown",
    }
    response = requests.post(url, data=payload)
    response_data = response.json()

    if not response_data["ok"]:
        raise Exception(f"Error updating message: {response_data}")
    return message_id


def send_photo(bot_token, chat_id, image_path, caption):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {"photo": open(image_path, "rb")}
    data = {"chat_id": chat_id, "caption": caption}
    response = requests.post(url, files=files, data=data)
    return response.json()


def format_table_as_markdown(headers, rows):
    """
    Formats a list of headers and rows into a Markdown table.
    """
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    return table
