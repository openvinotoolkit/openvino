from openvino.tools.mo.back.ie_ir_ver_2.emitter import append_ir_info
from openvino.tools.mo.utils.cli_parser import get_meta_info


def serialize(ngraph_function, xml_path, argv=None):
    from openvino.runtime import serialize  # pylint: disable=import-error,no-name-in-module
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
