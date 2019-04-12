"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
import json
from pathlib import Path
from argparse import ArgumentParser
from functools import partial

import numpy as np

from ..utils import get_path

from .format_converter import BaseFormatConverter


def build_argparser():
    parser = ArgumentParser(
        description="Converts annotation form a arbitrary format to accuracy-checker specific format", add_help=False
    )
    parser.add_argument(
        "converter",
        help="Specific converter to run",
        choices=list(BaseFormatConverter.providers.keys())
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="Directory to save converted annotation and meta info",
        required=False,
        type=partial(get_path, is_directory=True)
    )
    parser.add_argument("-m", "--meta_name", help="Meta info file name", required=False)
    parser.add_argument("-a", "--annotation_name", help="Annotation file name", required=False)
    parser.add_argument("-ss", "--subsample", help="Dataset subsample size", required=False)
    parser.add_argument("--subsample_seed", help="Seed for generation dataset subsample", type=int, required=False)

    return parser


def make_subset(annotation, size, seed=666):
    dataset_size = len(annotation)
    if dataset_size < size:
        warnings.warn('dataset size - {} less than subsample size - {}'.format(dataste_size, size))
        return annotation
    np.random.seed(seed)
    return list(np.random.choice(annotation, size=size, replace=False))


def main():
    main_argparser = build_argparser()
    args, _ = main_argparser.parse_known_args()
    converter, converter_argparser, converter_args = get_converter_arguments(args)

    main_argparser = ArgumentParser(parents=[main_argparser, converter_argparser])
    args = main_argparser.parse_args()

    converter = configure_converter(converter_args, args, converter)
    out_dir = args.output_dir or Path.cwd()

    result, meta = converter.convert()

    subsample = args.subsample
    if subsample:
        if subsample.endswith('%'):
            subsample_ratio = float(subsample[:-1]) / 100
            subsample_size = int(len(result) * subsample_ratio)
        else:
            subsample_size = int(args.subsample)

        result = make_subset(result, subsample_size)

    converter_name = converter.get_name()
    annotation_name = args.annotation_name or "{}.pickle".format(converter_name)
    meta_name = args.meta_name or "{}.json".format(converter_name)

    annotation_file = out_dir / annotation_name
    meta_file = out_dir / meta_name

    save_annotation(result, meta, annotation_file, meta_file)


def save_annotation(annotation, meta, annotation_file, meta_file):
    if annotation_file:
        with annotation_file.open('wb') as file:
            for representation in annotation:
                representation.dump(file)
    if meta_file and meta:
        with meta_file.open('wt') as file:
            json.dump(meta, file)


def configure_converter(converter_options, args, converter):
    args_dict, converter_options_dict = vars(args), vars(converter_options)
    converter_config = {
        option_name: option_value for option_name, option_value in args_dict.items()
        if option_name in converter_options_dict and option_value is not None
    }
    converter_config['converter'] = args.converter
    converter.config = converter_config
    converter.validate_config()
    converter.configure()

    return converter


def get_converter_arguments(arguments):
    converter = BaseFormatConverter.provide(arguments.converter)
    converter_argparser = converter.get_argparser()
    converter_options, _ = converter_argparser.parse_known_args()
    return converter, converter_argparser, converter_options


if __name__ == '__main__':
    main()
