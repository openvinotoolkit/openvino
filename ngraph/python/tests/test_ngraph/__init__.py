# ngraph.dll directory path visibility is needed to use _pyngraph module
# import below causes adding this path to os.environ["PATH"]
import ngraph  # noqa: F401 'imported but unused'
