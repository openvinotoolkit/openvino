#!/usr/bin/python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import glob
import ntpath
import re

class Code2CHeaders(object):
    def __init__(self, kernels_folder, headers_folder, lang):
        self.kernels_folder = os.path.abspath(kernels_folder)
        self.headers_folder = os.path.abspath(headers_folder)
        self.language = lang
        assert(self.language == "ocl" or self.language == "cm")

    def minimize_code(self, content):
        # Remove single-line comments (// ...)
        content = re.sub(r'//.*', '', content)

        # Remove multi-line comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Remove leading and trailing whitespaces from each line
        content = '\n'.join(line.strip() for line in content.splitlines())

        # Remove empty lines
        content = '\n'.join(line for line in content.splitlines() if line)

        # Normalize multiple spaces to a single space within lines
        content = '\n'.join(re.sub(r'\s+', ' ', line) for line in content.splitlines())

        # Remove unnecessary spaces/new lines around specific symbols
        content = '\n'.join(re.sub(r'\s*([{}=;,+\-<>!&|%#])\s*', r'\1', line) for line in content.splitlines())

        # Fold multi-line macros into single-line
        content = re.sub(r'\\\s*\n', '', content)

        return content

    # Replace include directives with header content if include is not batch header
    # For batch headers we just put a unique set of them in the beginning of the code
    # Note that includes at batch headers are not processed at this point and shall be handles in runtime
    def process_includes(self, content, include_paths, processed_includes=None):
        if processed_includes is None:
            processed_includes = set()

        batch_headers = []

        def replace_include(match):
            include_file = match.group(1)
            no_opt_attr = match.group(2) is not None
            if include_file in processed_includes and not no_opt_attr:
                return ''

            if "batch_headers" in include_file:
                processed_includes.add(include_file)
                batch_headers.insert(0, f'#include "{include_file}"')
            else:
                for path in include_paths:
                    full_path = os.path.join(path, include_file)
                    if os.path.isfile(full_path):
                        with open(full_path, 'r') as f:
                            included_content = f.read()
                        processed_includes.add(include_file)
                        return self.process_includes(included_content, include_paths, processed_includes)
            return ''

        content = re.sub(r'#include\s+"([^"]+)"(\s+\[\[no_opt\]\])?', replace_include, content)

        return '\n'.join(batch_headers) + '\n' + content.strip()

    def add_missing_undefs(self, content):
        # Find all #define directives (both regular and function-like macros)
        define_pattern = re.compile(r'#define\s+(\w+)\s*(?:\((.*?)\))?\s*')

        defines = set(match.group(1) for match in define_pattern.finditer(content))
        undef_directives = []
        content_lines = content.split("\n")
        for define in defines:
            # Regex to check if #undef for this define already exists
            undef_pattern = re.compile(r'#undef\s+' + re.escape(define) + r'\s*(?:\n|$)')

            if not undef_pattern.search(content):
                undef_directives.append(f'#undef {define}')

        if undef_directives:
            content += '\n' + '\n'.join(sorted(undef_directives))

        return content

    def remove_unused_macros(self, content):
        macro_pattern = re.compile(r'^#define\s+(\w+)\s*(.*)', re.MULTILINE)
        undef_pattern = re.compile(r'^#undef\s+(\w+)\b', re.MULTILINE)
        cat_pattern = re.compile(r'CAT\s*\(([^()]*|(?:[^()]*\([^\)]*\))[^()]*)\)')

        macros = {match.group(1): match.group(2) for match in macro_pattern.finditer(content)}

        used_macros = set()

        # Check for direct usage of macros (excluding their definition and undef lines)
        for macro in macros:
            if re.search(r'(?<!#define\s)(?<!#undef\s)\b' + macro + r'\b', content):
                used_macros.add(macro)

        # Expand CAT() recursively to track macro usage.
        # Note that it requires full macro name match
        # Maybe we'll need to relax that to deal with prefix matching?
        def expand_cat_expression(expression):
            # Repeatedly expand CAT expressions
            while True:
                matches = cat_pattern.findall(expression)
                if not matches:
                    break
                for match in matches:
                    parts = re.split(r'\s*,\s*', match)
                    expanded_name = ''.join(parts)
                    expression = expression.replace(f'CAT({match})', expanded_name)
            return expression

        def extract_macro_names(expression):
            expanded_expression = expand_cat_expression(expression)
            # Return all possible concatenation combinations, since any of them can result in used macros
            macro_names = set()
            if expanded_expression in macros:
                macro_names.add(expanded_expression)
            return macro_names

        # Track all used macros, including those generated by CAT()
        for macro in list(macros.keys()):
            used_macros.update(extract_macro_names(macros[macro]))

        # Ensure all macros used in CAT expressions are tracked
        for match in cat_pattern.finditer(content):
            used_macros.update(extract_macro_names(match.group()))

        # Find unused macros: those that are neither used directly nor indirectly
        unused_macros = set(macros.keys()) - used_macros

        # Ensure that if a macro is removed, its corresponding #undef is also removed
        def remove_macro_or_undef(match):
            return '' if match.group(1) in unused_macros else match.group(0)

        # Remove unused macros and undefs (if present)
        content = macro_pattern.sub(remove_macro_or_undef, content)
        content = undef_pattern.sub(remove_macro_or_undef, content)

        # Return content without extra blank lines
        return '\n'.join(line for line in content.splitlines() if line.strip())

    def process_file(self, filepath, include_dirs = [], is_batch_header = False):
        max_length = 5000
        filename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, 'r') as file:
            content = file.read()
            content = self.minimize_code(content)
            if not is_batch_header:
                content = self.process_includes(content, include_dirs)
                content = self.minimize_code(content)
                content = self.remove_unused_macros(content)
                content = self.add_missing_undefs(content)
            if len(content) > max_length:
                parts = [content[i:i+max_length] for i in range(0, len(content), max_length)]
                map_entry = '\n'.join([f'R"__krnl({part})__krnl"' for part in parts])
            else:
                map_entry = f'R"__krnl({content})__krnl"'

            processed_code = f'std::make_pair<std::string_view, std::string_view>("{filename}", {map_entry}),\n'

        return processed_code

    def generate(self):
        sources = []
        headers = []

        source_ext = {
            "ocl" : (".cl"),
            "cm" : (".cm")
        }
        headers_ext = {
            "ocl" : (".cl"),
            "cm" : (".h")
        }

        # Process kernel files
        for filename in sorted(os.listdir(self.kernels_folder)):
            if filename.endswith(source_ext[self.language]):
                filepath = os.path.join(self.kernels_folder, filename)
                print('processing {}'.format(filename))
                include_dirs = []
                include_dirs.append(self.kernels_folder)
                include_dirs.append(self.headers_folder)
                include_dirs.append(os.path.join(self.headers_folder, "../"))
                include_dirs.append(os.path.join(self.headers_folder, "batch_headers"))
                map_entry = self.process_file(filepath, include_dirs, is_batch_header=False)
                sources.append(map_entry)

        # Process batch header files in include directory
        include_dir = os.path.join(self.headers_folder, 'batch_headers')
        if os.path.exists(include_dir):
            for filename in sorted(os.listdir(include_dir)):
                if filename.endswith(headers_ext[self.language]):
                    filepath = os.path.join(include_dir, filename)
                    map_entry = self.process_file(filepath, is_batch_header=True)
                    headers.append(map_entry)

        return sources, headers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-in_kernels_dir', required=True, metavar='PATH', help='The absolute path to kernels folder')
    ap.add_argument('-in_headers_dir', required=True, metavar='PATH', help='The absolute path to headers root folder')
    ap.add_argument('-out_sources', required=True, metavar='PATH', help='The absolute path to output header with sources')
    ap.add_argument('-out_headers', required=True, metavar='PATH', help='The absolute path to output header with headers')
    ap.add_argument('-lang', required=True, help='Language of the source files. Supports `cm` and `ocl` for now')
    args = ap.parse_args()

    converter = Code2CHeaders(args.in_kernels_dir, args.in_headers_dir, args.lang)
    kernel_entries, header_entries = converter.generate()

    def write_to_file(file_path, content : list):
        with open(file_path, 'w') as f:
            for entry in content:
                f.write(entry)

    write_to_file(args.out_sources, kernel_entries)
    write_to_file(args.out_headers, header_entries)

if __name__ == '__main__':
    main()
