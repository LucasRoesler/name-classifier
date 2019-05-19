import string

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)
N_HIDDEN = 128

# Build the category_lines dictionary, a list of names per language

ALL_CATEGORIES = [
    "German",
    "English",
    "Czech",
    "Portuguese",
    "Japanese",
    "Polish",
    "Chinese",
    "Scottish",
    "Spanish",
    "Irish",
    "French",
    "Italian",
    "Russian",
    "Vietnamese",
    "Greek",
    "Arabic",
    "Dutch",
    "Korean",
]

N_CATEGORIES = len(ALL_CATEGORIES)
