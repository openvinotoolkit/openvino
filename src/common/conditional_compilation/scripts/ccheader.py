#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#     The main purpose of this script is code generation for conditional compilation.
# After collecting statistics using IntelSEAPI, several CSV files are generated.
# This script can read these files and can produce header file which will contain
# definitions for enabled OpenVINO parts.
#
#     Usage: ccheader.py [-h] --stat PATH[PATH...] [PATH[PATH...] ...] --out cc.h
#
#     Mandatory arguments:
#   --stat PATH[ PATH...] [PATH[ PATH...] ...]
#                         IntelSEAPI statistics files in CSV format
#   --out cc.h            C++ header file to be generated

import argparse, csv
from glob import glob
from pathlib import Path
from abc import ABC, abstractmethod

Domain = ["SIMPLE_", "SWITCH_", "FACTORY_", "TYPE_LIST_"]

FILE_HEADER = "#pragma once\n\n"
FILE_FOOTER = "\n"

ENABLED_SCOPE_FMT = "#define %s_%s 1\n"
ENABLED_SWITCH_FMT = "#define %s_%s 1\n#define %s_%s_cases %s\n"
ENABLED_FACTORY_INSTANCE_FMT = "#define %s_%s 1\n"

class IScope(ABC):
    @abstractmethod
    def generate(self, f, module):
        pass


class Scope(IScope):
    def __init__(self, name):
        self.name = name

    def generate(self, f, module):
        f.write(ENABLED_SCOPE_FMT % (module, self.name))

class Switch(IScope):
    def __init__(self, name):
        self.name = name
        self.cases = set()

    def case(self, val):
        self.cases.add(val)

    def generate(self, f, module):
        f.write(ENABLED_SWITCH_FMT % (module, self.name, module, self.name, ', '.join(self.cases)))

class Factory(IScope):
    def __init__(self, name):
        self.name = name
        self.registered = {}
        self.created = set()

    def register(self, id, name):
        self.registered[id] = name

    def create(self, id):
        self.created.add(id)

    def generate(self, f, module):
        for id in self.created:
            r = self.registered.get(id)
            if r:
                f.write(ENABLED_FACTORY_INSTANCE_FMT % (module, r))


class TypeList(IScope):
    et_sep = ", "
    et_namespace = "::ov::element::"

    def __init__(self, name):
        self.name = name
        self.types = set()

    def append(self, type):
        self.types.add(type)

    def generate(self, f, module):
        type_list = self.et_sep.join((self.et_namespace + t for t in self.types))
        f.write(f"#define {module}_enabled_{self.name} 1\n")
        f.write(f"#define {module}_{self.name} {type_list}\n")


class Module:
    def __init__(self, name):
        self.name = name
        self.scopes = {}

    def scope(self, name):
        if name not in self.scopes:
            self.scopes[name] = Scope(name)
        return self.scopes.get(name)

    def factory(self, name):
        if name not in self.scopes:
            self.scopes[name] = Factory(name)
        return self.scopes.get(name)

    def switch(self, name):
        if name not in self.scopes:
            self.scopes[name] = Switch(name)
        return self.scopes.get(name)

    def generate(self, f):
        for _, scope in self.scopes.items():
            scope.generate(f, self.name)
        if self.scopes:
            f.write("\n")

    def type_list(self, scope_name):
        return self.scopes.setdefault(scope_name, TypeList(scope_name))


class Stat:
    def __init__(self, files):
        self.modules = {}
        self.read(files)

    def module(self, name):
        if name not in self.modules:
            self.modules[name] = Module(name)
        return self.modules.get(name)

    def read(self, files):
        for stats in files:
            for stat in glob(str(stats)):
                with open(str(stat)) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if rows:
                        # Scopes
                        scopes = list(filter(lambda row: len(row) and row[0].startswith(Domain[0]), rows))
                        for row in scopes:
                            moduleName = row[0][len(Domain[0]):]
                            self.module(moduleName).scope(row[1])

                        # Switches
                        switches = list(map(lambda row: [row[0][len(Domain[1]):]] + row[1].strip().split('$'),
                                            filter(lambda row: len(row) and row[0].startswith(Domain[1]), rows)))
                        for switch in switches:
                            self.module(switch[0]).switch(switch[1]).case(switch[2])

                        # Factories
                        factories = list(map(lambda row: [row[0][len(Domain[2]):]] + row[1].strip().split('$'),
                                            filter(lambda row: len(row) and row[0].startswith(Domain[2]), rows)))
                        for reg in list(filter(lambda row: len(row) > 1 and row[1] == 'REG', factories)):
                            self.module(reg[0]).factory(reg[2]).register(reg[3], reg[4])
                        for cre in list(filter(lambda row: len(row) > 1 and row[1] == 'CREATE', factories)):
                            self.module(cre[0]).factory(cre[2]).create(cre[3])

                        # Type list generator filter, returns tuple of (domain, (region, type))
                        type_list_filter = (
                            (row[0], row[1].strip().split("$"))
                            for row in rows
                            if len(row) > 1 and row[0].startswith(Domain[3])
                        )

                        for domain, (region, type) in type_list_filter:
                            module = self.module(domain)
                            module.type_list(region).append(type)

    def generate(self, out):
        with open(str(out), 'w') as f:
            f.write(FILE_HEADER)

            for _, module in self.modules.items():
                module.generate(f)

            f.write(FILE_FOOTER)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stat', type=Path, nargs='+', metavar='PATH[ PATH...]',
        help='IntelSEAPI statistics files in CSV format', required=True)
    parser.add_argument('--out', type=Path, metavar='cc.h',
        help='C++ header file to be generated', required=True)
    args = parser.parse_args()

    stat = Stat(args.stat)
    stat.generate(args.out)

if __name__ == '__main__':
    main()
