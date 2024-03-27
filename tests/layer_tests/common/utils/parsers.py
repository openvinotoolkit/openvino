# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import defusedxml.ElementTree as ET


def mapping_parser(file):
    """
    Parse mapping file if it exists
    :param file: Name of mapping file
    :return: Dictionary with framework layers as keys and IR layers as values
    """
    mapping_dict = {}
    if os.path.splitext(file)[1] == '.mapping' and os.path.isfile(file):
        xml_tree = ET.parse(file)
        xml_root = xml_tree.getroot()
        for child in xml_root:
            framework_info = child.find('.//framework')
            ir_info = child.find('.//IR')
            if framework_info is None:
                continue
            framework_name = framework_info.attrib['name']
            ir_name = ir_info.attrib['name'] if ir_info is not None else None
            mapping_dict[framework_name] = ir_name
    else:
        raise FileNotFoundError("Mapping file was not found at path {}!".format(os.path.dirname(file)))
    return mapping_dict
