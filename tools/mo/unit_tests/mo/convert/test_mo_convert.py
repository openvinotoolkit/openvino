import sys
import os
import subprocess

from openvino.tools.mo.subprocess_main import setup_env, subprocess_main


def test_legacy_extensions():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'legacy_extensions_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode