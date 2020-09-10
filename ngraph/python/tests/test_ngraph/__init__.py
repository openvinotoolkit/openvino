import os
import sys
from pathlib import Path

if sys.platform == "win32":
    # ngraph.dll is installed in PYTHON_EXE_DIR/Lib as default
    # and this path needs to be visible to the _pyngraph module.
    #
    # If you're using a custom installation of nGraph,
    # add the location of ngraph.dll to your system PATH.
    ngraph_dll_dir = os.path.join(Path(sys.executable).parent, "Lib")
    os.environ["PATH"] = os.path.abspath(ngraph_dll_dir) + ";" + os.environ["PATH"]
