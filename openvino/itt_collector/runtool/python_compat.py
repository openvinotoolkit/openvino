import sys

if sys.version_info[0] > 2:
    import queue as queue
    basestring = (str, bytes)
    unicode = str

    def func_name(func_object):
        return func_object.__name__
    def func_globals(func_object):
        return func_object.__globals__
    def func_code(func_object):
        return func_object.__code__

    raw_input = input
else:
    import Queue as queue
    basestring = basestring
    unicode = unicode

    def func_name(func_object):
        return func_object.func_name
    def func_globals(func_object):
        return func_object.func_globals
    def func_code(func_object):
        return func_object.func_code
