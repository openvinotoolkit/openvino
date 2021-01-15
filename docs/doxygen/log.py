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

import argparse
import os
import re

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, default=None, help='Path to doxygen log file')
    parser.add_argument('--ignore-list', type=str, required=False,
                        default=os.path.join(os.path.abspath(os.path.dirname(__file__)),'doxygen-ignore.txt'),
                        help='Path to doxygen ignore list')
    parser.add_argument('--strip', type=str, required=False, default=os.path.abspath('../../'),
                        help='Strip from warning paths')
    parser.add_argument('--include_omz', type=bool, required=False, default=False,
                        help='Include link check for omz docs')
    parser.add_argument('--include_wb', type=bool, required=False, default=False,
                        help='Include link check for workbench docs')
    parser.add_argument('--include_pot', type=bool, required=False, default=False,
                        help='Include link check for pot docs')
    parser.add_argument('--include_gst', type=bool, required=False, default=False,
                        help='Include link check for gst docs')
    return parser.parse_args()


def strip_path(path, strip):
    """Strip `path` components ends on `strip`
    """
    path = path.replace('\\', '/')
    if path.endswith('.md') or path.endswith('.tag'):
        strip = os.path.join(strip, 'build/docs').replace('\\', '/') + '/'
    else:
        strip = strip.replace('\\', '/') + '/'
    return path.split(strip)[-1]


def is_excluded_link(warning, exclude_links):
    if 'unable to resolve reference to' in warning:
        ref = re.findall(r"'(.*?)'", warning)
        if ref:
            ref = ref[0]
            for link in exclude_links:
                reg = re.compile(link)
                if re.match(reg, ref):
                    return True
    return False


def parse(log, ignore_list, strip, include_omz=False, include_wb=False, include_pot=False, include_gst=False):
    found_errors = []
    exclude_links = {'omz': r'.*?omz_.*?', 'wb': r'.*?workbench_.*?',
                     'pot': r'.*?pot_.*?', 'gst': r'.*?gst_.*?'}
    if include_omz:
        del exclude_links['omz']
    if include_wb:
        del exclude_links['wb']
    if include_pot:
        del exclude_links['pot']
    if include_gst:
        del exclude_links['gst']

    exclude_links = exclude_links.values()

    with open(ignore_list, 'r') as f:
        ignore_list = f.read().splitlines()
    with open(log, 'r') as f:
        log = f.read().splitlines()
    for line in log:
        if 'warning:' in line:
            path, warning = list(map(str.strip, line.split('warning:')))
            path, line_num = path[:-1].rsplit(':', 1)
            path = strip_path(path, strip)
            if path in ignore_list or is_excluded_link(warning, exclude_links):
                continue
            else:
                found_errors.append('{path} {warning} line: {line_num}'.format(path=path,
                                                                               warning=warning,
                                                                               line_num=line_num))
    if found_errors:
        print('\n'.join(found_errors))
        exit(1)


def main():
    args = parse_arguments()
    parse(args.log,
          args.ignore_list,
          args.strip,
          include_omz=args.include_omz,
          include_wb=args.include_wb,
          include_pot=args.include_pot,
          include_gst=args.include_gst)


if __name__ == '__main__':
    main()
