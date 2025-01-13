"""
 Copyright (C) 2018-2025 Intel Corporation
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
import os
import sys
import re
import logging as log

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

def parse_hello_reshape_ssd(stdout):
    """
    This function get stdout from hello_reshape_ssd (already splitted by new line)
    The test check not if resulted class of object is accurate with reference, but that demo detected class with its box
    and so on and so forth.
    Checks:
    1) Resulting input shape = [1,3,500,500]
    The test checks that line includes text 'Resulting input shape = ' with box in format: [1,3,500,500]
    2) Resulting output shape = [1,1,200,7]
    The test checks that line includes text 'Resulting output shape = ' with box in format: [1,1,200,7]
    3) [33,59] element, prob = 0.963015, bbox = (189.776,110.933)-(309.288,306.952), batch id = 0
    The test checks that line includes text 'prob = ' with probability, 'bbox' with box in format:
    (189.776,110.933)-(309.288,306.952). Then text 'batch id =' with number.

    :param stdout: stdout from hello_reshape_ssd (already splitted by new line)
    :return: True or False as a result of check
    """
    is_ok = True
    for line in stdout:
        if 'Resulting input shape' in line:
            if re.match(r"^Resulting input shape\s+=\s+\[\d+,\d+,\d+,\d+]", line) is None:
                is_ok = False
                log.error('Wrong output line: {}, while the test expects the following format: 4d shape'
                          '(Example: Resulting input shape = [1,3,500,500])'.format(line))
        elif 'Resulting output shape' in line:
            if re.match(r"^Resulting output shape\s+=\s+\[\d+,\d+,\d+,\d+]", line) is None:
                is_ok = False
                log.error('Wrong output line: {}, while the test expects the following format: 4d shape'
                          '(Example: Resulting output shape = [1,1,200,7])'.format(line))
        elif 'element, prob' in line:
            if re.match("^.*prob\\s+=.*\\d,\\s+\\(.*\\d,.*\\d\\)-\\(.*\\d,.*\\d\\)", line) is None:
                is_ok = False
                log.error('Wrong output line: {}, while the test expects the following format: 4d shape'
                          '(Example: [33,59] element, prob = 0.963015, bbox = (189.776,110.933)-(309.288,306.952), '
                          'batch id = 0)'.format(line))
        elif 'was saved' in line:
            path_result = line.split(' ')[-1].strip()
            if not os.path.isfile(path_result):
                log.error("Image after infer was not found: {}".format(path_result))
                is_ok = False
    return is_ok
