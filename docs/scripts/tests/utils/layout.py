# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
DoxygenLayout.xml parsing routines
"""

from collections import defaultdict
import argparse
from lxml import etree
import re


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, required=True, default=None, help='Path to DoxygenLayout.xml file')
    return parser.parse_args()


def format_input(root):
    """
    Format input
    """
    for elem in root.getiterator():
        if not hasattr(elem.tag, 'find'):
            continue
        elem.tag = re.sub(r'{.+}(.+)', r'\1', elem.tag)


def parse_layout(content):
    """
    Parse DoxygenLayout.xml
    """
    parser = etree.XMLParser(encoding='utf-8')
    root = etree.fromstring(content, parser=parser)
    format_input(root)
    files = defaultdict(lambda: set())
    md_links = filter(
        lambda x: 'url' in x.attrib and x.attrib['url'].startswith('./') and x.attrib['url'].endswith('.md'),
        root.xpath('//tab'))
    for md_link in map(lambda x: x.attrib['url'], md_links):
        link = md_link[2:] if md_link.startswith('./') else md_link
        files[link] = set()
        files[link].update(["The link to this file located in DoxygenLayout.xml is not converted to a doxygen reference ('@ref filename')"])
    return files


if __name__ == '__main__':
    arguments = parse_arguments()
    with open(arguments.layout, 'r', encoding="utf-8") as f:
        content = f.read()
    parse_layout(content)
