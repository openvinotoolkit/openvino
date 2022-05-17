#!/usr/bin/env python
#===============================================================================
# Copyright 2021 Intel Corporation
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
#===============================================================================

from __future__ import print_function

import os
import re
import sys


def is_special_tag(t):
    return 'undef' in t or 'any' in t or 'tag_last' in t


def is_abc_tag(t):
    if is_special_tag(t):
        return False
    t = t.replace('dnnl_', '')
    for c in t:
        c = c.lower()
        if c.isdigit():
            continue
        if c.isalpha():
            if 'a' <= c and c <= 'l':
                continue
        return False
    return True


class Tag:
    def __init__(self, line):
        m = re.match(r'\s*(\w+)\s*=\s*(\w+)', line)
        if m:
            self.lhs = m.group(1)
            self.rhs = m.group(2)
        else:
            m = re.match(r'\s*(\w+)', line)
            self.lhs = m.group(1)
            self.rhs = None

        self.is_special = is_special_tag(self.lhs)
        self.is_abc = is_abc_tag(self.lhs)
        if self.is_special:
            self.rhs = None
        elif not self.rhs:
            assert self.is_abc, ('Expected abc-tag: %s' % line)

    def lhs_base_tag(self):
        for s in ['undef', 'any', 'last']:
            if s in self.lhs:
                return s
        return self.lhs.replace('dnnl_', '')

    def rhs_base_tag(self):
        return self.rhs.replace('dnnl_', '')

    def __str__(self):
        return str((self.lhs, self.rhs))


def usage():
    print('''\
Usage: %s

Updates dnnl.hpp header with missing format tags from dnnl_types.h''' %
          sys.argv[0])
    sys.exit(1)


for arg in sys.argv:
    if '-help' in arg:
        usage()

script_root = os.path.dirname(os.path.realpath(__file__))

dnnl_types_h_path = '%s/../include/oneapi/dnnl/dnnl_types.h' % script_root
dnnl_hpp_path = '%s/../include/oneapi/dnnl/dnnl.hpp' % script_root

c_tags = []
cpp_tags = []

# Parse tags from dnnl_types.h
with open(dnnl_types_h_path) as f:
    s = f.read()
    m = re.search(r'.*enum(.*?)dnnl_format_tag_t', s, re.S)
    lines = [l for l in m.group(1).split('\n') if l.strip().startswith('dnnl')]
    for l in lines:
        c_tags.append(Tag(l))

# Parse tags from dnnl.hpp
with open(dnnl_hpp_path) as f:
    dnnl_hpp_contents = f.read()
    m = re.search(r'(enum class format_tag.*?)};', dnnl_hpp_contents, re.S)
    dnnl_hpp_format_tag = m.group(1)
    lines = [
        l for l in dnnl_hpp_format_tag.split('\n')
        if l.strip() and '=' in l.strip()
    ]
    for l in lines:
        cpp_tags.append(Tag(l))

# Validate dnnl.hpp tags
for cpp_tag in cpp_tags:
    if cpp_tag.is_special:
        continue
    if cpp_tag.rhs:
        if cpp_tag.lhs_base_tag() == cpp_tag.rhs_base_tag():
            continue
        tags = [t for t in c_tags if t.lhs_base_tag() == cpp_tag.lhs_base_tag()]
        if tags:
            if cpp_tag.rhs_base_tag() == tags[0].rhs_base_tag():
                continue
        print('Can\'t validate tag: %s' % cpp_tag)

# Find missing aliases in dnnl.hpp
missing_dnnl_hpp_tag_lines = []
for c_tag in c_tags:
    if c_tag.is_special:
        continue
    cpp_found = [
        t for t in cpp_tags if t.lhs_base_tag() == c_tag.lhs_base_tag()
    ]
    if not cpp_found:
        base = c_tag.lhs_base_tag()
        line = '        %s = dnnl_%s,' % (base, base)
        missing_dnnl_hpp_tag_lines.append(line)

if not missing_dnnl_hpp_tag_lines:
    exit(0)

old_dnnl_hpp_format_tag = dnnl_hpp_format_tag
dnnl_hpp_format_tag = dnnl_hpp_format_tag.rstrip()
dnnl_hpp_format_tag += '\n' + '\n'.join(missing_dnnl_hpp_tag_lines) + '\n    '

dnnl_hpp_contents = dnnl_hpp_contents.replace(old_dnnl_hpp_format_tag,
                                              dnnl_hpp_format_tag)

# Update dnnl.hpp
with open(dnnl_hpp_path, 'w') as f:
    f.write(dnnl_hpp_contents)
