import os
from contextlib import contextmanager


@contextmanager
def environment_update(env_vars):
    old_env = dict(os.environ)
    sets = {k: v for k, v in env_vars.items() if v is not None}
    unsets = [k for k, v in env_vars.items() if v is None]
    os.environ.update(sets)
    for var in unsets:
        os.environ.pop(var, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)
