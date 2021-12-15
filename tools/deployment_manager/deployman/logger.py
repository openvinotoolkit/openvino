"""
 Copyright (c) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log


def init_logger(lvl: str):
    # create logger
    logger = log.getLogger('DeploymentManager')
    logger.setLevel(lvl)

    # create console handler and set level to debug
    ch = log.StreamHandler()
    ch.setLevel(log.DEBUG)

    # create formatter
    formatter = log.Formatter('[ %(asctime)s ] %(levelname)s : %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger

