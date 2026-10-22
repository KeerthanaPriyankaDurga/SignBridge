def prepare_speech_text(text: str, language: str = "en") -> dict:
    cleaned = " ".join(text.split())

    if language not in {"en", "te"}:
        raise ValueError("Supported languages: en, te")

    return {
        "text": cleaned,
        "language": language,
        "ready": bool(cleaned),
    }


if __name__ == "__main__":
    result = prepare_speech_text("HELLO THANK YOU", "en")
    print(result)
