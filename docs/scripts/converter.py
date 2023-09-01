# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import re
import shutil

from pathlib import Path


INLINE_LINKS_PATTERN = r'!?\[.*?\]\(([\w\/\-\.]+\.md)\)'
REFERENCE_LINKS_PATTERN = r'\[.+\]\:\s*?([\w\/\-\.]+\.md)'
INLINE_IMAGES_PATTERN = r'!?\[.*?\]\(([\w\/\-\.]+\.(?:png|jpg|jpeg|gif|svg))\)'
REFERENCE_IMAGES_PATTERN = r'\[.+\]\:\s*?([\w\/\-\.]+\.(?:png|jpg|jpeg|gif|svg))'
LABEL_PATTERN = r'\{\#(.+)\}'
DEFAULT_EXCLUDES = '^.*(thirdparty|bin|tests|temp|docs/ops/internal).*$'

class RSTHelper:
    def __init__(self, rst_file, label, input_dir, output_dir):
        self.rst_file = rst_file
        self.label = label
        self.parent_folder = self.get_parent_folder()
        self.content = self.get_content()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_links = self.get_image_links()

    def get_parent_folder(self):
        """
        Get parent folder of a rst file
        """
        return self.rst_file.parents[0]

    def get_content(self):
        """
        Read rst file
        """
        with open(self.rst_file, 'r', encoding='utf-8') as f:
            return f.read()

    def replace_image_links(self):
        """
        Replace image links
        """
        for image in self.image_links:
            self.content = self.content.replace(image, self.parent_folder.joinpath(image).name)

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
        dest = self.output_dir.joinpath(path.name)
        try:
            shutil.copy(path, dest)
        except FileNotFoundError:
            logging.warning('{}: image not found'.format(path))

    def copy(self):
        """
        Save processed content of rst file in output_dir
        """
        output = os.path.join(self.output_dir, f'{self.label}.rst')
        with open(output, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def run(self):
        """
        Do all processing operations on a rst file
        """
        self.replace_image_links()
        self.copy_images()
        self.copy()

    def get_image_links(self):
        """
        Get image links from the page
        """
        inline_images = set(re.findall(INLINE_IMAGES_PATTERN, self.content, flags=re.IGNORECASE))
        reference_images = set(re.findall(REFERENCE_IMAGES_PATTERN, self.content, flags=re.IGNORECASE))
        image_links = inline_images
        image_links.update(reference_images)
        return image_links


def get_label(file):
    """
    Read lines of a file and try to find a label.
    If the label is not found return None.
    Assume the label is in the first line
    :return: A label
    """
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        label = re.search(LABEL_PATTERN, line)
        if label:
            return label.group(1)
    

def filter_paths(rst_files, exclude_dirs):
    """
    Exclude paths presented in exclude_dirs
    :param rst_files: A list of rst files
    :param exclude_dirs: A list of directories to be excluded from processing
    :return: A list of rst files to be process after excluded paths
    """
    #rst_files = filter(lambda x: not re.match(DEFAULT_EXCLUDES, str(x.parent)), rst_files)
    return filter(lambda x: not any(ex_path in x.parents for ex_path in exclude_dirs), rst_files)


def process(input_dir, output_dir, exclude_dirs):
    """
    Recursively find rst files and process them accordingly
    """
    rst_files = input_dir.glob('**/*.rst')
    rst_files = filter_paths(rst_files, exclude_dirs)
    for rst_file in rst_files:
        label = get_label(rst_file)
        if label and not re.match(DEFAULT_EXCLUDES, str(rst_file.parent)):
            logging.info("Processing {} file".format(rst_file))
            helper = RSTHelper(rst_file=rst_file,
                               label=label,
                               input_dir=input_dir,
                               output_dir=output_dir)
            helper.run()
        else:
            logging.info("Skipped {} file - no label or matched excludes".format(rst_file))


def main():
    logging.basicConfig(level=logging.INFO)
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
