import re

def xor(a, b):
  return bool(a) ^ bool(b)

def deg2rad(deg):
  pi_on_180 = 0.017453292519943295
  return deg * pi_on_180

def get_valid_filename(file_name):
  file_name = re.sub(r'[()]', '', file_name)
  file_name = re.sub(r'\s', '-', file_name)

  return file_name
