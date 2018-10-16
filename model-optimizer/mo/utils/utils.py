"""
 Copyright (c) 2018 Intel Corporation

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


def refer_to_faq_msg(question_num: int):
    return '\n For more information please refer to Model Optimizer FAQ' \
           ' (<INSTALL_DIR>/deployment_tools/documentation/docs/MO_FAQ.html),' \
           ' question #{}. '.format(question_num)


class NamedAttrsClass:
    def __init__(self, class_attrs: dict):
        for key, val in class_attrs.items():
            self.__setattr__(key, val)
