import os

from openvino.tools.mo.back.ie_ir_ver_2.emitter import append_ir_info
from openvino.tools.mo.utils.cli_parser import get_meta_info
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.telemetry_utils import get_tid
from openvino.tools.mo.utils.utils import refer_to_faq_msg
from openvino.tools.mo.utils.version import get_simplified_mo_version

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


def serialize(ngraph_function, xml_path, argv=None):
    from openvino.runtime import serialize  # pylint: disable=import-error,no-name-in-module
    telemetry = tm.Telemetry(tid=get_tid(), app_name='Model Optimizer', app_version=get_simplified_mo_version())
    telemetry.start_session('mo')
    telemetry.send_event('mo', 'version', get_simplified_mo_version())

    output_dir = os.path.dirname(xml_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except PermissionError as e:
            raise Error("Failed to create directory {}. Permission denied! " +
                        refer_to_faq_msg(22),
                        output_dir) from e
    else:
        if not os.access(output_dir, os.W_OK):
            raise Error("Output directory {} is not writable for current user. " +
                        refer_to_faq_msg(22), output_dir)

    serialize(ngraph_function, xml_path.encode('utf-8'), xml_path.replace('.xml', '.bin').encode('utf-8'))

    if argv is not None:
        from openvino.offline_transformations import \
            generate_mapping_file  # pylint: disable=import-error,no-name-in-module

        path_to_mapping = xml_path.replace('.xml', '.mapping').encode('utf-8')
        extract_names = argv.framework in ['tf', 'mxnet', 'kaldi']
        generate_mapping_file(ngraph_function, path_to_mapping, extract_names)

        # add MO params information to IR
        append_ir_info(file=xml_path,
                       meta_info=get_meta_info(argv),
                       mean_data=None,
                       input_names=None,
                       legacy_path=False)

    telemetry.end_session('mo')
