# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime

msg_fmt = 'It\'s been a while, check for a new version of ' + \
          'Intel(R) Distribution of OpenVINO(TM) toolkit here {0} or on the GitHub*'


def get_ov_update_message():
    expected_update_date = datetime.date(year=2022, month=4, day=30)
    current_date = datetime.date.today()

    link = 'https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2022_bu_IOTG_OpenVINO-2022-1&content=upg_all&medium=organic'

    return msg_fmt.format(link) if current_date >= expected_update_date else None


def get_ov_api20_message():
    link = "https://docs.openvino.ai/latest/documentation.html"
    message = '[ INFO ] Starting from 2022.1 release the Model Optimizer generates IR in a format compatible with OpenVINO(TM) API 2.0. ' \
              'OpenVINO(TM) API 2.0 provides better alignment with original frameworks such as model inputs and outputs format. ' \
              'For more information about OpenVINO(TM) API 2.0 please follow: {}'. format(link)

    return message
