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

            matches = re.findall(pattern, contents, flags=re.DOTALL)
            if matches:
                for match in matches:
                    contents = contents.replace(match, saxutils.escape(match))
            # escape asterisks
            contents = contents.replace('*', '\\*')
            contents = str.encode(contents)
            root = etree.fromstring(contents)

            # unescape * in sphinxdirectives
            sphinxdirectives = root.xpath('//sphinxdirective')
            for sd in sphinxdirectives:
                sd_text = sd.text.replace('\\*', '*')
                sd.text = sd_text
            anchors = root.xpath('//anchor')
            localanchors = list(filter(lambda x: x.attrib['id'].startswith('_1'), anchors))
            for anc in localanchors:
                text = anc.attrib['id']
                heading = anc.getparent()
                para = heading.getparent()
                dd = para.getparent()
                index = dd.index(para)
                new_para = etree.Element('para')
                sphinxdirective_anchor = etree.Element('sphinxdirective')
                sphinxdirective_anchor.text = '\n\n.. _' + text[2:] + ':\n\n'
                new_para.append(sphinxdirective_anchor)
                dd.insert(index, new_para)

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
