import torch
from FormatSSVAE.const import ALL_LETTERS, PAD_CHAR

def pad_string(original: str, desired_len: int, pad_character: str = PAD_CHAR):
    """
    Returns the padded version of the original string to length: desired_len
    original: The string to be padded
    """
    return (pad_character * (desired_len - len(original))) + original

def chars_to_tensor(chars: list, letter_set: str):
    """
    Turn a list of name strings into a tensor of one-hot letter vectors
    of shape: <max string length x len(strings) x len(letter_set)>
    """
    tensor = torch.zeros(1, len(chars), len(letter_set))
    for i_letter, letter in enumerate(chars):
        letter_index = letter_set.find(letter)
        if letter_index == -1: raise Exception(f'Invalid Character {letter} Detected')
        tensor[0][i_letter][letter_index] = 1
    return tensor

def strings_to_tensor(strings: list, letter_set: str, tensor_len: int):
    """
    Turn a list of name strings into a tensor of one-hot letter vectors
    of shape: <max string length x len(strings) x num letters>

    All names are padded with '<pad_character>' such that they have the length: desired_len
    names: List of names to converted to a one-hot-encded vector
    max_name_len: The max name length allowed
    """
    tensor = torch.zeros(tensor_len, len(strings), len(letter_set))
    padded_strs = list(map(lambda s: pad_string(s, tensor_len), strings))
    for i_string, string in enumerate(padded_strs):
        for i_char, letter in enumerate(string):
            letter_index = letter_set.find(letter)
            if letter_index == -1: raise Exception(f'Invalid Character {letter} Detected')
            tensor[i_char][i_string][letter_index] = 1
    return tensor

