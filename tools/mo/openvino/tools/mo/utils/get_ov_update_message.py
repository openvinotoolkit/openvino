# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime

msg_fmt = 'Check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here {0} ' \
          'or on https://github.com/openvinotoolkit/openvino'


def get_ov_update_message():
    expected_update_date = datetime.date(year=2024, month=12, day=1)
    current_date = datetime.date.today()

    link = 'https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2023_bu_IOTG_OpenVINO-2023-1&content=upg_all&medium=organic'

    return msg_fmt.format(link) if current_date >= expected_update_date else None


def get_compression_message():
    link = "https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html"
    message = '[ INFO ] Generated IR will be compressed to FP16. ' \
              'If you get lower accuracy, please consider disabling compression explicitly ' \
              'by adding argument --compress_to_fp16=False.\n' \
              'Find more information about compression to FP16 at {}'.format(link)
    return message


def get_try_legacy_fe_message():
    message = '[ INFO ] You can also try to use legacy TensorFlow Frontend by using argument --use_legacy_frontend.\n'
    return message


def get_ovc_message():
    link = "https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html"
    message = '[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.\n' \
              'In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. ' \
              'Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). ' \
              'OVC represents a lightweight alternative of MO and provides simplified model conversion API. \n' \
              'Find more information about transition from MO to OVC at {}'.format(link)

    return message
