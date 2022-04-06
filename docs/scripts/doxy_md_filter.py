# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import argparse
from pathlib import Path
import shutil
import logging

INLINE_LINKS_PATTERN = r'!?\[.*?\]\(([\w\/\-\.]+\.md)\)'
REFERENCE_LINKS_PATTERN = r'\[.+\]\:\s*?([\w\/\-\.]+\.md)'
INLINE_IMAGES_PATTERN = r'!?\[.*?\]\(([\w\/\-\.]+\.(?:png|jpg|jpeg|gif|svg))\)'
REFERENCE_IMAGES_PATTERN = r'\[.+\]\:\s*?([\w\/\-\.]+\.(?:png|jpg|jpeg|gif|svg))'
LABEL_PATTERN = r'\{\#(.+)\}'


class DoxyMDFilter:
    def __init__(self, md_file, input_dir, output_dir, file_to_label_mapping):
        self.md_file = md_file
        self.file_to_label_mapping = file_to_label_mapping
        self.parent_folder = self.get_parent_folder()
        self.content = self.get_content()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_links = self.get_image_links()
        self.md_links = self.get_md_links()

    def copy_markdown(self):
        """
        Save processed content of markdown file in output_dir
        """
        rel_path = self.md_file.relative_to(self.input_dir)
        dest = self.output_dir.joinpath(rel_path)
        dest.parents[0].mkdir(parents=True, exist_ok=True)
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def get_parent_folder(self):
        """
        Get parent folder of a markdown file
        """
        return self.md_file.parents[0]

    def get_content(self):
        """
        Read Markdown File
        """
        with open(self.md_file, 'r', encoding='utf-8') as f:
            return f.read()

    def replace_image_links(self):
        """
        Replace relative image links with absolute paths.
        This is needed for doxygen.
        """
        for image in self.image_links:
            new_path = self.parent_folder.joinpath(image).resolve().relative_to(self.input_dir)
            new_path = self.output_dir / new_path
            self.content = self.content.replace(image, new_path.as_posix())

    def replace_md_links(self):
        """
        Replace markdown links with doxygen labels.
        """
        for link in self.md_links:
            link_path = self.parent_folder.joinpath(link).resolve()
            if os.path.exists(link_path) and link_path in self.file_to_label_mapping:
                self.content = self.content.replace(link, '@ref ' + self.file_to_label_mapping[link_path])
            else:
                rel_path = os.path.relpath(link_path, self.input_dir).replace('\\', '/')
                self.content = self.content.replace(link, rel_path)

    def remove_comment_block_sphinxdirective(self):
        """
        Remove comment blocks from `sphinxdirective`
        """
        self.content = re.sub(r'\<\!\-\-\s*?\@sphinxdirective', '@sphinxdirective', self.content)
        self.content = re.sub(r'\@endsphinxdirective\s*?\-\-\>', '@endsphinxdirective', self.content)

    def copy_images(self):
        """
        Go through image links and copy them into output_folder
        """
        for image in self.image_links:
            path = self.parent_folder.joinpath(image)
            self._copy_image(path)

    def _copy_image(self, path):
        """
        A helper function for self.copy_images
        """
        path = path.resolve()
        rel_path = path.relative_to(self.input_dir)
        dest = self.output_dir.joinpath(rel_path)
        dest.parents[0].mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(path, dest)
        except FileNotFoundError:
            logging.warning('{}: image not found'.format(path))

    def filter(self):
        """
        Do all processing operations on a markdown file
        """
        self.replace_image_links()
        self.remove_comment_block_sphinxdirective()
        self.replace_md_links()
        self.copy_markdown()
        self.copy_images()

    def get_image_links(self):
        """
        Get image links from the page
        """
        inline_images = set(re.findall(INLINE_IMAGES_PATTERN, self.content, flags=re.IGNORECASE))
        reference_images = set(re.findall(REFERENCE_IMAGES_PATTERN, self.content, flags=re.IGNORECASE))
        image_links = inline_images
        image_links.update(reference_images)
        return image_links

    def get_md_links(self):
        """
        Get markdown links from the page
        """
        inline_links = set(re.findall(INLINE_LINKS_PATTERN, self.content))
        reference_links = set(re.findall(REFERENCE_LINKS_PATTERN, self.content))
        md_links = inline_links
        md_links.update(reference_links)
        return sorted(md_links, key=len, reverse=True)


def get_label(file):
    """
    Read lines of a file and try to find a doxygen label.
    If the label is not found return None.
    Assume the label is in the first line
    :return: A doxygen label
    """
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        label = re.search(LABEL_PATTERN, line)
        if label:
            return label.group(1)


def get_file_to_label_mapping(md_files):
    """
    Get a dictionary containing path as keys and
    doxygen labels as values
    :param md_files: A list of md files
    :return: A dictionary containing a file to label mapping 
    """
    return dict(filter(lambda x: x[1], map(lambda f: (Path(f), get_label(f)), md_files)))
    

def filter_paths(md_files, exclude_dirs):
    """
    Exclude paths presented in exclude_dirs
    :param md_files: A list of md files
    :param exclude_dirs: A list of directories to be excluded from processing
    :return: A list of md files to be process after excluded paths
    """
    return filter(lambda x: not any(ex_path in x.parents for ex_path in exclude_dirs), md_files)


def process(input_dir, output_dir, exclude_dirs):
    """
    Recursively find markdown files in docs_folder and
    replace links to markdown files with doxygen labels (ex. @ref label_name).
    """
    md_files = input_dir.glob('**/*.md')
    md_files = filter_paths(md_files, exclude_dirs)
    file_to_label_map = get_file_to_label_mapping(md_files)
    for md_file in file_to_label_map.keys():
        dmdf = DoxyMDFilter(md_file=md_file,
                            input_dir=input_dir,
                            output_dir=output_dir,
                            file_to_label_mapping=file_to_label_map)
        dmdf.filter()


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
