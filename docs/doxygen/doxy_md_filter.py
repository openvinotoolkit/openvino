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

import os
import re
import glob
import argparse


def get_label(file):
    """
    Read lines of a file and try to find a doxygen label.
    If the label is not found return None.
    """
    for line in file:
        label = re.search(r'\{\#(.+)\}', line)
        if label:
            return label.group(1)
    return


def replace_links(content, items, folder, labels, docs_folder):
    """
    Replace markdown links with doxygen labels.
    """
    for item in items:
        link = item[0]
        ext = item[1]
        link_path = os.path.abspath(os.path.join(folder, link))
        if os.path.exists(link_path):
            if ext == 'md':
                if link_path in labels:
                    label = labels.get(link_path)
                else:
                    with open(link_path, 'r', encoding='utf-8') as file:
                        lines = []
                        i = 0
                        while i < 5:
                            try:
                                lines.append(next(file))
                            except StopIteration:
                                break
                            i += 1
                        label = get_label(lines)
                        labels[link_path] = label
                if label:
                    content = content.replace(link, '@ref ' + label)
            else:
                rel_path = os.path.relpath(link_path, docs_folder).replace('\\', '/')
                content = content.replace(link, rel_path)
    return content


def process_github_md_links(content, items):
    """
    This is a workaround to support github markdown links in doxygen 1.8.12.
    """
    for item in items:
        orig = item[0]
        link_name = item[1]
        link_url = item[2]
        html_link = '<a href="{}">{}</a>'.format(link_url, link_name)
        content = content.replace(orig, html_link)
    return content


def process(docs_folder):
    """
    Recursively find markdown files in docs_folder and
    replace links to markdown files with doxygen labels (ex. @ref label_name).
    """
    labels = dict()  # store labels in dictionary
    md_files = glob.glob(os.path.join(docs_folder, '**/*.md'), recursive=True)
    for md_file in md_files:
        md_folder = os.path.dirname(md_file)
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        inline_links = set(re.findall(r'!?\[.*?\]\(([\w\/\-\.]+\.(md|png|jpg|gif))\)', content, flags=re.IGNORECASE))
        github_md_links = set(re.findall(r'(\[(.+?)\]\((https:[\w\.\/-]+?\.md)\))', content, flags=re.IGNORECASE))
        reference_links = set(re.findall(r'\[.+\]\:\s*?([\w\/\-\.]+\.(md|png|jpg|gif))', content, flags=re.IGNORECASE))
        content = replace_links(content, inline_links, md_folder, labels, docs_folder)
        content = replace_links(content, reference_links, md_folder, labels, docs_folder)
        content = process_github_md_links(content, github_md_links)
        if inline_links or reference_links or github_md_links:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docs', type=str, help='Path to a folder containing .md files.')
    args = parser.parse_args()
    process(args.docs)


if __name__ == '__main__':
    main()
