# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import logging
import argparse
from lxml import etree
from pathlib import Path
from xml.sax import saxutils


def prepare_xml(xml_dir: Path):
    """
    A preprocessing xml function
    """
    pattern = r'\<sphinxdirective\>(.+?)\<\/sphinxdirective>'
    xml_files = xml_dir.glob('*.xml')

    for xml_file in xml_files:
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                contents = f.read()

            sphinxdirectives = re.findall(pattern, contents, flags=re.DOTALL)
            if sphinxdirectives:
                for sd in sphinxdirectives:
                    contents = contents.replace(sd, saxutils.escape(sd))
            contents = str.encode(contents)
            root = etree.fromstring(contents)
            anchors = root.xpath('//anchor')
            localanchors = list(filter(lambda x: x.attrib['id'].startswith('_1'), anchors))
            for anc in localanchors:
                text = anc.attrib['id']
                heading = anc.getparent()
                para = heading.getparent()
                dd = para.getparent()
                index = dd.index(para)
                new_para = etree.Element('para')
                sphinxdirective = etree.Element('sphinxdirective')
                sphinxdirective.text = '\n\n.. _' + text[2:] + ':\n\n'
                new_para.append(sphinxdirective)
                dd.insert(index, new_para)

            # memberdefs
            # memberdefs = root.xpath('//memberdef')
            # for member in memberdefs:
            #     name = member.find('name').text
            #     id = member.attrib['id'].replace('__', '_').replace('_1_1_', '_').replace('_1_1', '_')
            #     id = '_'.join(id.split('_')[:-1]) + '_' + name
            #     member.attrib['id'] = id.lower()
            #     _type = member.find('type')

            with open(xml_file, 'wb') as f:
                f.write(etree.tostring(root))
        except UnicodeDecodeError as err:
            logging.warning('{}:{}'.format(xml_file, err))


def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_dir', type=Path, help='Path to the folder containing xml files.')
    args = parser.parse_args()
    xml_dir = args.xml_dir
    prepare_xml(xml_dir)


if __name__ == '__main__':
    main()
