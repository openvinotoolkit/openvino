import os
import sys
import ngraph as ng

if sys.platform == "win32":
    # ngraph.dll is installed in PYTHON_INSTALL_DIR/Lib as default
    # (which is 2 directories above ngraph module)
    # and this path needs to be visible to the _pyngraph module.
    #
    #
    # If you're using a custom installation of nGraph,
    # add the location of ngraph.dll to your system PATH.
    ngraph_dll_dir = os.path.join(os.path.dirname(ng.__file__), "..", "..")
    os.environ["PATH"] = os.path.abspath(ngraph_dll_dir) + ";" + os.environ["PATH"]
