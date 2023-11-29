# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm
from openvino.tools.ovc.convert_impl import _convert
from openvino.tools.ovc.cli_parser import get_model_name_from_args

# pylint: disable=no-name-in-module,import-error
from openvino.runtime import save_model


def main():
    from openvino.tools.ovc.cli_parser import get_all_cli_parser
    ngraph_function, argv = _convert(get_all_cli_parser(), {}, False)
    if ngraph_function is None:
        return 1

    model_path = get_model_name_from_args(argv)

    compress_to_fp16 = 'compress_to_fp16' in argv and argv.compress_to_fp16
    save_model(ngraph_function, model_path.encode('utf-8'), compress_to_fp16)

    print('[ SUCCESS ] XML file: {}'.format(model_path))
    print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
    return 0


if __name__ == "__main__":
    sys.exit(main())
