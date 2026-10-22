import unittest
from src.sentence_builder.builder import SentenceBuilder


class TestSentenceBuilder(unittest.TestCase):
    def test_add_letters(self):
        builder = SentenceBuilder()
        for letter in "HELLO":
            builder.add_letter(letter)
        self.assertEqual(builder.get_text(), "HELLO")

    def test_add_word(self):
        builder = SentenceBuilder()
        for letter in "HELLO":
            builder.add_letter(letter)
        builder.add_word("THANK YOU")
        self.assertEqual(builder.get_text(), "HELLO THANK YOU")

    def test_clear(self):
        builder = SentenceBuilder()
        builder.add_word("HELLO")
        builder.clear()
        self.assertEqual(builder.get_text(), "")


if __name__ == "__main__":
    unittest.main()
