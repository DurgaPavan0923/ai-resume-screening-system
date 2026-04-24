def highlight_text(text, keywords):
    for word in keywords:
        text = text.replace(
            word,
            f"<span style='background-color:yellow; color:black'>{word}</span>"
        )
    return text
