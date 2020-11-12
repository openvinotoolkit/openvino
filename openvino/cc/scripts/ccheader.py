#!/usr/bin/env python3

import argparse, csv
from pathlib import Path

Domain = ['CC0_',
          'CC1_',
          'CC2_']

FILE_HEADER = "#pragma once\n\n"

FILE_FOOTER = "\n"

ENABLED_SCOPE_FMT = "#define %s_%s 1\n"
ENABLED_SWITCH_FMT = "#define %s_%s 1\n#define %s_%s_cases %s\n"
ENABLED_FACTORY_INSTANCE_FMT = "#define %s_%s 1\n"

class Scope:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def generate(self, f, module):
        f.write(ENABLED_SCOPE_FMT % (module, self.name))

class Switch:
    def __init__(self, name):
        self.name = name
        self.cases = set()

    def case(self, val):
        self.cases.add(val)

    def generate(self, f, module):
        f.write(ENABLED_SWITCH_FMT % (module, self.name, module, self.name, ', '.join(self.cases)))

class Factory:
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
        if self.created:
            f.write("\n")

class Module:
    def __init__(self, name):
        self.name = name
        self.scopes = set()
        self.switches = {}
        self.factories = {}

    def scope(self, name):
        self.scopes.add(Scope(name))

    def factory(self, name):
        if name not in self.factories:
            self.factories[name] = Factory(name)
        return self.factories.get(name)

    def switch(self, name):
        if name not in self.switches:
            self.switches[name] = Switch(name)
        return self.switches.get(name)

    def generate(self, f):
        for scope in self.scopes:
            scope.generate(f, self.name)
        if self.scopes:
            f.write("\n")

        for _, switch in self.switches.items():
            switch.generate(f, self.name)
        if self.switches:
            f.write("\n")

        for _, factory in self.factories.items():
            factory.generate(f, self.name)
        if self.factories:
            f.write("\n")

class Stat:
    def __init__(self, files):
        self.modules = {}
        self.read(files)

    def module(self, name):
        if name not in self.modules:
            self.modules[name] = Module(name)
        return self.modules.get(name)

    def read(self, files):
        for stat in files:
            with open(stat) as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    # Scopes
                    scopes = list(filter(lambda row: row[0].startswith(Domain[0]), rows))
                    for row in scopes:
                        moduleName = row[0][len(Domain[0]):]
                        self.module(moduleName).scope(row[1])

                    # Switches
                    switches = list(map(lambda row: [row[0][len(Domain[1]):]] + row[1].strip().split('$'),
                                        filter(lambda row: row[0].startswith(Domain[1]), rows)))
                    for switch in switches:
                        self.module(switch[0]).switch(switch[1]).case(switch[2])

                    # Factories
                    factories = list(map(lambda row: [row[0][len(Domain[2]):]] + row[1].strip().split('$'),
                                        filter(lambda row: row[0].startswith(Domain[2]), rows)))
                    for reg in list(filter(lambda row: row[1] == 'REG', factories)):
                        self.module(reg[0]).factory(reg[2]).register(reg[3], reg[4])
                    for cre in list(filter(lambda row: row[1] == 'CREATE', factories)):
                        self.module(cre[0]).factory(cre[2]).create(cre[3])

    def generate(self, out):
        with open(out, 'w') as f:
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
