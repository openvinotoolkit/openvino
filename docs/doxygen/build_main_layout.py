# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import argparse
import os
from lxml import etree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openvino', type=str, required=True, default=None, help='openvino_docs.xml')
    args = parser.parse_args()
    result = build_layout(args.openvino)
    with open(args.openvino, 'wb') as f:
        f.write(result)


def build_layout(openvino):
    ns = {"xi": "http://www.w3.org/2001/XInclude"}
    root = etree.parse(openvino)
    root.xinclude()
    return etree.tostring(root, pretty_print=True)


if __name__ == '__main__':
    main()
