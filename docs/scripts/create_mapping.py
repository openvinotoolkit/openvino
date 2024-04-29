# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
from lxml import etree
from pathlib import Path

REPOSITORIES = [
    'openvino',
    'omz',
    'pot',
    'ovms',
    'ote',
]


def create_mapping(xml_input: Path, output_dir: Path, strip_path: Path):
    """
    Create a mapping between doxygen label and file path for edit on github button.
    """
    xml_input = xml_input.resolve()
    output_dir = output_dir.resolve()
    strip_path = strip_path.resolve()
    mapping = {
        'get_started': 'openvino/docs/get_started.md',
        'ovsa_get_started': 'openvino/docs/ovsa/ovsa_get_started.md',
        'documentation': 'openvino/docs/documentation.md',
        'index': 'openvino/docs/index.rst',
        'model_zoo': 'openvino/docs/model_zoo.md',
        'resources': 'openvino/docs/resources.md',
        'tutorials': 'openvino/docs/tutorials.md',
        'tuning_utilities': 'openvino/docs/tuning_utilities.md'
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_files = xml_input.glob('*.xml')
    for xml_file in xml_files:
        try:
            root = etree.parse(xml_file.as_posix()).getroot()
            compounds = root.xpath('//compounddef')
            for compound in compounds:
                kind = compound.attrib['kind']
                if kind in ['file', 'dir']:
                    continue
                name_tag = compound.find('compoundname')
                name = name_tag.text
                name = name.replace('::', '_1_1')
                if kind == 'page':
                    exclude = True
                    for rep in REPOSITORIES:
                        if name.startswith(rep):
                            exclude = False
                    if exclude:
                        continue
                else:
                    name = kind + name
                location_tag = compound.find('location')
                file = Path(location_tag.attrib['file'])
                if not file.suffix:
                    continue
                try:
                    file = file.relative_to(strip_path)
                except ValueError:
                    logging.warning('{}: {} is not relative to {}.'.format(xml_file, file, strip_path))
                mapping[name] = file.as_posix()
        except AttributeError:
            logging.warning('{}: Cannot find the origin file.'.format(xml_file))
        except etree.XMLSyntaxError as e:
            logging.warning('{}: {}.'.format(xml_file, e))

    with open(output_dir.joinpath('mapping.json'), 'w') as f:
        json.dump(mapping, f)


def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_input', type=Path, help='Path to the folder containing doxygen xml files')
    parser.add_argument('output_dir', type=Path, help='Path to the output folder')
    parser.add_argument('strip_path', type=Path, help='Strip from path')
    args = parser.parse_args()
    xml_input = args.xml_input
    output_dir = args.output_dir
    strip_path = args.strip_path
    create_mapping(xml_input, output_dir, strip_path)


if __name__ == '__main__':
    main()
