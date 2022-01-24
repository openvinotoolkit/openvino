# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import argparse
from pathlib import Path
import shutil
import logging


def get_label(file):
    """
    Read lines of a file and try to find a doxygen label.
    If the label is not found return None.
    Assume the label is in the first line
    """
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        label = re.search(r'\{\#(.+)\}', line)
        if label:
            return label.group(1)


def replace_links(content, items, md_folder, labels, docs_folder):
    """
    Replace markdown links with doxygen labels.
    """
    for item in items:
        link = item
        link_path = md_folder.joinpath(link).resolve()
        if os.path.exists(link_path):
            content = content.replace(link, '@ref ' + labels[link_path])
        else:
            rel_path = os.path.relpath(link_path, docs_folder).replace('\\', '/')
            content = content.replace(link, rel_path)
    return content


def replace_image_links(content, images, input_dir, md_folder, output_dir):
    for image in images:
        new_path = md_folder.joinpath(image).resolve().relative_to(input_dir)
        new_path = output_dir / new_path
        content = content.replace(image, new_path.as_posix())
    return content


def add_htmlonly(content):
    content = content.replace('<details>', '\n\\htmlonly\n<details>')
    content = content.replace('</summary>', '</summary>\n\\endhtmlonly')
    content = content.replace('</details>', '\n\\htmlonly\n</details>\n\\endhtmlonly')
    content = content.replace('<iframe', '\n\\htmlonly\n<iframe')
    content = content.replace('</iframe>', '</iframe>\n\\endhtmlonly')
    return content


def copy_file(file, content, input_dir, output_dir):
    rel_path = file.relative_to(input_dir)
    dest = output_dir.joinpath(rel_path)
    dest.parents[0].mkdir(parents=True, exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)


def copy_image(file, input_dir, output_dir):
    rel_path = file.relative_to(input_dir)
    dest = output_dir.joinpath(rel_path)
    dest.parents[0].mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(file, dest)
    except FileNotFoundError:
        logging.warning('{}: file not found'.format(file))


def get_refs_by_regex(content, regex):
    def map_func(path):
        return (Path(path[0]), path[1]) if isinstance(path, tuple) else Path(path)

    refs = set(map(map_func, re.findall(regex, content, flags=re.IGNORECASE)))
    return refs


def process(input_dir, output_dir, exclude_dirs):
    """
    Recursively find markdown files in docs_folder and
    replace links to markdown files with doxygen labels (ex. @ref label_name).
    """
    md_files = input_dir.glob('**/*.md')
    md_files = filter(lambda x: not any(ex_path in x.parents for ex_path in exclude_dirs), md_files)
    label_to_file_map = dict(filter(lambda x: x[1], map(lambda f: (Path(f), get_label(f)), md_files)))
    label_to_file_map.pop(None, None)
    for md_file in label_to_file_map.keys():
        md_folder = md_file.parents[0]
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        inline_links = set(re.findall(r'!?\[.*?\]\(([\w\/\-\.]+\.md)\)', content))
        reference_links = set(re.findall(r'\[.+\]\:\s*?([\w\/\-\.]+\.md)', content))
        inline_images = set(re.findall(r'!?\[.*?\]\(([\w\/\-\.]+\.(?:png|jpg|gif|svg))\)', content, flags=re.IGNORECASE))
        reference_images = set(re.findall(r'\[.+\]\:\s*?([\w\/\-\.]+\.(?:png|jpg|gif|svg))', content, flags=re.IGNORECASE))

        images = inline_images
        images.update(reference_images)

        content = replace_image_links(content, images, input_dir, md_folder, output_dir)

        md_links = inline_links
        md_links.update(reference_links)
        md_links = list(filter(lambda x: md_folder.joinpath(x) in label_to_file_map, md_links))
        content = replace_links(content, md_links, md_folder, label_to_file_map, input_dir)
        # content = add_htmlonly(content)

        copy_file(md_file, content, input_dir, output_dir)

        for image in images:
            path = md_file.parents[0].joinpath(image)
            copy_image(path, input_dir, output_dir)


def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, help='Path to a folder containing .md files.')
    parser.add_argument('--output_dir', type=Path, help='Path to the output folder.')
    parser.add_argument('--exclude_dir', type=Path, action='append', default=[], help='Ignore a folder.')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    exclude_dirs = args.exclude_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    process(input_dir, output_dir, exclude_dirs)


if __name__ == '__main__':
    main()
