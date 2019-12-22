import string

MAX_STRING_LEN = 20

EOS_CHAR = '0'
PAD_CHAR = '1'
DROP_CHAR = '2'
ALL_LETTERS = string.ascii_lowercase + " .,'-" + EOS_CHAR + PAD_CHAR
OUTPUT_LETTERS = ALL_LETTERS + DROP_CHAR