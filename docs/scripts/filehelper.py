# Copyright (C) 2018-2025 Intel Corporation
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
DEFAULT_EXCLUDES = {
    'rst': r'^.*(thirdparty|bin|tests|temp|docs/ops/internal).*$',
    'md': r'$-'  # to not match anything
}
                    

class FileHelper:
    def __init__(self, filetype, file, label, input_dir, output_dir, file_to_label_mapping):
        self.filetype = filetype
        self.file = file
        self.label = label
        self.parent_folder = self.get_parent_folder()
        self.content = self.get_content()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_links = self.get_image_links()
        if self.filetype == "md":
            self.md_links = self.get_md_links()
        self.file_to_label_mapping=file_to_label_mapping

    def get_parent_folder(self):
        """
        Get parent folder of a file
        """
        return self.file.parents[0]

    def get_content(self):
        """
        Read file
        """
        with open(self.file, 'r', encoding='utf-8') as f:
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
        Save processed content of file in output_dir
        """
        output = os.path.join(self.output_dir, f'{self.label}.{self.filetype}')
        with open(output, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def replace_md_links(self):
        """
        Replace markdown links with doxygen labels.
        """
        for link in self.md_links:
            link_path = self.parent_folder.joinpath(link).resolve()
            if os.path.exists(link_path) and link_path in self.file_to_label_mapping:
                self.content = self.content.replace(link, self.file_to_label_mapping[link_path] + '.md')

    def remove_comment_block_sphinxdirective(self):
        """
        Remove comment blocks from `sphinxdirective`
        """
        self.content = re.sub(r'\<\!\-\-\s*?\@sphinxdirective', '@sphinxdirective', self.content)
        self.content = re.sub(r'\@endsphinxdirective\s*?\-\-\>', '@endsphinxdirective', self.content)

    def remove_label(self):
        """
        Remove label from processed md file.
        """
        label_regex = r'\{\#' + re.escape(self.label) + r'\}'
        self.content = re.sub(label_regex, '', self.content)

    def run(self):
        """
        Do all processing operations on a file
        """
        self.replace_image_links()
        if self.filetype == "md":
            self.replace_md_links()
            self.remove_label()
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


def get_file_to_label_mapping(md_files):
    """
    Get a dictionary containing path as keys and
    doxygen labels as values
    :param md_files: A list of md files
    :return: A dictionary containing a file to label mapping 
    """
    return dict(filter(lambda x: x[1], map(lambda f: (Path(f).resolve(), get_label(f)), md_files)))


def filter_paths(files, exclude_dirs):
    """
    Exclude paths presented in exclude_dirs
    :param files: A list of files
    :param exclude_dirs: A list of directories to be excluded from processing
    :return: A list of files to be process after excluded paths
    """
    return filter(lambda x: not any(ex_path in x.parents for ex_path in exclude_dirs), files)


def process(filetype, input_dir, output_dir, exclude_dirs):
    """
    Recursively find files and process them accordingly
    """
    files = input_dir.glob("**/*.{}".format(filetype))
    files = filter_paths(files, exclude_dirs)
    file_to_label_map = get_file_to_label_mapping(files)
    for file in file_to_label_map.keys():
        label = get_label(file)
        if label and not re.match(DEFAULT_EXCLUDES[filetype], str(file.parent)):
            logging.info("Processing {} file".format(file))
            helper = FileHelper(filetype=filetype,
                                file=file,
                                label=label,
                                input_dir=input_dir,
                                output_dir=output_dir,
                                file_to_label_mapping=file_to_label_map)
            helper.run()
        else:
            logging.info("Skipped {} file - no label or matched excludes".format(file))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--filetype', choices=['md', 'rst'], help='Type of processed files, allowed md or rst.')
    parser.add_argument('--input_dir', type=Path, help='Path to a folder containing files.')
    parser.add_argument('--output_dir', type=Path, help='Path to the output folder.')
    parser.add_argument('--exclude_dir', type=Path, action='append', default=[], help='Ignore a folder.')
    args = parser.parse_args()
    filetype = args.filetype
    input_dir = args.input_dir
    output_dir = args.output_dir
    exclude_dirs = args.exclude_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    process(filetype, input_dir, output_dir, exclude_dirs)


if __name__ == '__main__':
    main()
