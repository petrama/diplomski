import importlib.util
import time

def import_module(name, path):
  print(path)
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

def get_expired_time(start_time):
    curr_time = time.time()
    delta = curr_time - start_time
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds