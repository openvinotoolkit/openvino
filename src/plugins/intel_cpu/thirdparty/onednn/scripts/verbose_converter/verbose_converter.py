#!/usr/bin/env python
################################################################################
# Copyright 2020-2021 Intel Corporation
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
################################################################################

import sys

import argparse
from argparse import RawTextHelpFormatter

from src import utils
from src import writer

def convert(verbose_level, parser, input, action, generator, split_output):
    status = utils.check_version()
    if status != utils.status.get('SUCCESS'):
        return status

    logger = writer.Writer(verbose_level=verbose_level)
    log_parser = None
    if parser == 'oneDNN':
        from src import dnnl_parser
        log_parser = dnnl_parser.LogParser(logger, input)
    else:
        logger.print("Error: unsupported parser", 'STDIO')
        return utils.status.get('FAILED')

    logger.print(f'Processing input ...', 'INFO')
    log_parser.process()

    output = None
    if action == 'dumpIR':
        logger.print(f'Dumping data from input...', 'INFO')
        log_parser.dump(True)
    if action == 'generate':
        if generator == 'benchdnn':
            from src import benchdnn_generator
            gen = benchdnn_generator.InputGenerator(logger)
        else:
            logger.print("Error: unsupported generator", 'STDIO')
            return utils.status.get('FAILED')

        logger.print(f'Generating output ...', 'INFO')
        output = gen.generate(log_parser.get_data(), split_output)
    return utils.status.get('SUCCESS'), output

def validate_option(value, supported_values, str):
    if not value in supported_values:
        print(f"ERROR: {str}")
        return utils.status.get('FAILED')
    return utils.status.get('SUCCESS')

def main():
    status = utils.check_version()
    if status != utils.status.get('SUCCESS'):
        return status

    action_opts = ["generate", "dumpIR"]
    generator_opts = ["benchdnn"]
    parser_opts = ["oneDNN"]
    verbose_opts = ["0", "1"]

    args_parser = argparse.ArgumentParser(description='oneDNN log converter',
                                          formatter_class=RawTextHelpFormatter)
    args_parser.add_argument('-i',
                             '--input',
                             default='stdin',
                             help='input file (default: stdin)')
    args_parser.add_argument(
        '-p',
        '--parser',
        default='oneDNN',
        help=f'type of parser (default: oneDNN). Values: {parser_opts}.')
    args_parser.add_argument(
        '-a',
        '--action',
        default='generate',
        help=f'an action (default: generate). Values: {action_opts}.')
    args_parser.add_argument(
        '-s',
        '--split',
        type=bool,
        default=False,
        help='split generated inputs by primitive kinds (default: False)')
    args_parser.add_argument(
        '-v',
        '--verbose_level',
        default='0',
        help=f'verbose level (default: 0). Values: {verbose_opts}.')
    args_parser.add_argument('-o',
                             '--output',
                             default='stdout',
                             help='output file (default: stdout)')
    args_parser.add_argument(
        '-g',
        '--generator',
        default='benchdnn',
        help=f'target generator (default: benchdnn). Values: {generator_opts}.')
    args = args_parser.parse_args()

    # validate options
    status = validate_option(args.action, action_opts, "Unknown action value")
    if status != utils.status.get('SUCCESS'):
        return status
    status = validate_option(args.verbose_level, verbose_opts, "Unknown verbose_level value")
    if status != utils.status.get('SUCCESS'):
        return status
    status = validate_option(args.parser, parser_opts, "Unknown parser value")
    if status != utils.status.get('SUCCESS'):
        return status
    status = validate_option(args.generator, generator_opts, "Unknown generator value")
    if status != utils.status.get('SUCCESS'):
        return status

    input_data = []
    if args.input == 'stdin':
        # if no input was piped, skip reading
        if not sys.stdin.isatty():
            for line in sys.stdin:
                input_data.append(line)
        else:
            print("WARN: no input was provided to the script")
    else:
        try:
            input_data = open(args.input, 'r').readlines()
        except BaseException as e:
            print(f"Error while reading input: {e}")

    output = None

    status, output = convert(verbose_level=args.verbose_level,
                             parser=args.parser,
                             input=input_data,
                             action=args.action,
                             generator=args.generator,
                             split_output=args.split)

    if status != utils.status.get('SUCCESS'):
        return status

    if output != None:
        if args.output != 'stdout':
            if output != None:
                for key, value in output.items():
                    filename = args.output
                    if args.split == True:
                        filename += '.' + key
                    of = open(filename, 'w')
                    print(value, end='', file=of)
        else:
            for key, value in output.items():
                if args.split == False:
                    print(f"{value}")
                else:
                    print(f"--{key}\n{value}")


if __name__ == "__main__":
    main()
