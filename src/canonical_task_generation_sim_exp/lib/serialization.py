import numpy as np
import ast
import re

def from_np_array(array_string):
    array_starts = [i for i in range(len(array_string)) if array_string.startswith("array(", i)]
    arrays = []
    for s in array_starts:
        substr = array_string[s + len("array(["):]
        arr = substr[:substr.find("]")]
        arrays.append([float(x) for x in arr.split(',')])
    return arrays

def from_list(list_str):
    list_str = list_str.replace('  ', ',').replace('\r\n ', ',')
    return ast.literal_eval(list_str)

def from_space_sep_list(list_str):
    return ast.literal_eval(','.join(re.sub(r'(?<=\d)(\s+)(?=-?\d)', ',', list_str).splitlines()))