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

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker, IRawChecker


class BackslashChecker(BaseChecker):
    """
    Checks for line continuations with '\' instead of using triple quoted string or parenthesis.
    """

    __implements__ = IRawChecker

    name = 'backslash'
    msgs = {
        'W9901': (
            'use of \\ for line continuation', 'backslash-line-continuation',
            'Used when a \\ is used for a line continuation instead of using triple quoted string or parenthesis.'
        ),
    }
    options = ()

    def process_module(self, node):
        with node.stream() as stream:
            for (line_number, line) in enumerate(stream):
                if not line.decode().rstrip().endswith('\\'):
                    continue

                self.add_message('backslash-line-continuation', line=line_number)


class AbsoluteImportsChecker(BaseChecker):
    """
    Check for absolute import from the same package.
    """

    __implements__ = IAstroidChecker

    name = 'absolute-imports'
    priority = -1
    msgs = {
        'W9902': (
            'absolute import from same package', 'package-absolute-imports',
            'Used when module of same package imported using absolute import'
        )
    }

    def visit_importfrom(self, node):
        node_package = self._node_package(node)
        import_name = node.modname
        if import_name.startswith(node_package):
            self.add_message('package-absolute-imports', node=node)

    @staticmethod
    def _node_package(node):
        return node.scope().name.split('.')[0]


class StringFormatChecker(BaseChecker):
    """
    Check for absolute import from the same package.
    """

    __implements__ = IAstroidChecker

    name = 'string-format'
    priority = -1
    msgs = {
        'W9903': (
            'use of "%" for string formatting', 'deprecated-string-format',
            '"%" operator is used for string formatting instead of str.format method'
        )
    }

    def visit_binop(self, node):
        if node.op != '%':
            return

        left = node.left
        if not (isinstance(left, astroid.Const) and isinstance(left.value, str)):
            return

        self.add_message('deprecated-string-format', node=node)


class BadFunctionChecker(BaseChecker):
    """
    Check for absolute import from the same package.
    """

    __implements__ = IAstroidChecker

    name = 'bad-function'
    priority = -1
    msgs = {'W9904': ('using prohibited function', 'bad-function-call', '')}

    options = (
        (
            'bad-functions',
            {
                'default': '',
                'help': 'List of prohibited functions',
            },
        ),
    )

    def visit_call(self, node):
        bad_functions = set(f.strip() for f in self.config.bad_functions.split(','))
        if self._function_name(node) in bad_functions:
            self.add_message('bad-function-call', node=node)

    @staticmethod
    def _function_name(node):
        func = node.func
        if hasattr(func, 'attrname'):
            return func.attrname
        elif hasattr(func, 'name'):
            return func.name


def register(linter):
    """
    Required method to auto register this checker.
    """

    linter.register_checker(BackslashChecker(linter))
    linter.register_checker(AbsoluteImportsChecker(linter))
    linter.register_checker(StringFormatChecker(linter))
    linter.register_checker(BadFunctionChecker(linter))
