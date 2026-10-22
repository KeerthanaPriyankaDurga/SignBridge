
class SentenceBuilder:
    def __init__(self):
        self.text = ""

    def add_letter(self, letter):
        letter = str(letter).strip()
        if len(letter) == 1 and letter.isalpha():
            self.text += letter.upper()

    def add_digit(self, digit):
        digit = str(digit).strip()
        if len(digit) == 1 and digit.isdigit():
            self.text += digit

    def add_word(self, word):
        word = " ".join(str(word).split()).upper()
        if not word:
            return

        if self.text and not self.text.endswith(" "):
            self.text += " "

        self.text += word

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def add_punctuation(self, mark):
        if mark in {".", ",", "?", "!", "'"} and self.text:
            if self.text.endswith(" "):
                self.text = self.text.rstrip()
            self.text += mark

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""

    def set_text(self, text):
        self.text = str(text)

    def get_text(self):
        return self.text


if __name__ == "__main__":
    builder = SentenceBuilder()

    for letter in "HELLO":
        builder.add_letter(letter)

    builder.add_word("THANK YOU")
    print("Output:", builder.get_text())

    builder.add_space()
    builder.add_digit("5")
    print("After digit:", builder.get_text())

    builder.backspace()
    print("After backspace:", builder.get_text())
