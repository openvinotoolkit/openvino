from openvino.utils import _add_openvino_libs_to_search_path

_add_openvino_libs_to_search_path()

from openvino.frontend.pytorch.torchdynamo import backend