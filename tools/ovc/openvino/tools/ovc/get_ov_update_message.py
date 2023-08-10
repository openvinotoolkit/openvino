# Copyright (C) 2018-2023 Intel Corporation
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
              'If you get lower accuracy, please consider disabling compression ' \
              'by removing argument "compress_to_fp16" or set it to false "compress_to_fp16=False".\n' \
              'Find more information about compression to FP16 at {}'.format(link)
    return message
