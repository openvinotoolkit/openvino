import re
import argparse


def process_pyx(pyx_file):
    """
    Convert .pyx file to a more readable format for doxygen.
    """
    with open(pyx_file, 'r') as f:
        source = f.readlines()
    idx = 0
    while idx < len(source):
        line = source[idx]
        striped_line = line.lstrip()
        tabs = ' ' * (len(line) - len(striped_line))  # Keep indentation
        striped_line = striped_line.rstrip()
        if striped_line == '@property':  # Python functions wrapped with @property decorator
            new_getter = convert_getter(source, idx)
            if new_getter:
                indent = tabs + ' ' * 4
                new_func, comments, shift = new_getter
                func_name = re.search(r'def\s+?([A-Za-z0-9_]+)\s*?\(', new_func).group(1)
                source[idx + 1] = tabs + new_func + '\n'
                for i in range(shift):
                    source.pop(idx + 2)
                # This is a workaround to help Doxygen understand "@property" functions as class properties.
                for comm in comments:
                    source.insert(idx + 2, '{indent}{comment}\n'.format(indent=indent, comment=comm))
                    idx += 1
                source.insert(idx + 2, '{indent}self.{func_name} = {func_name}\n'.format(
                    indent=indent,
                    func_name=func_name
                ))
                idx += 1
        if re.search(r'c?p?def.+\(', striped_line):  # Convert cython functions to python format
            new_sign = get_signature(source, idx)
            if new_sign:
                new_func, shift = new_sign
                args = re.search(r'\((.+)\)', new_func)
                if args:
                    new_func = new_func.replace(args.group(1), process_args(args.group(1))).replace('cpdef', 'def')
                source[idx] = tabs + new_func + '\n'
                for i in range(shift):
                    source.pop(idx + 1)
        if '__cinit__' in striped_line:  # Doxygen only interprets "__init__" constructors
            source[idx] = source[idx].replace('__cinit__', '__init__')
        idx += 1

    with open(pyx_file, 'w') as f:
        f.writelines(source)


def process_args(str_args):
    """
    Convert function arguments to the doxygen readable format.
    """
    args = re.sub(r'\[.*?\]', r'', str_args)
    args = re.sub(r'\(.*?\)', r'', args)
    args = args.split(',')
    for idx, arg in enumerate(args):
        arg = arg.replace('&', '').strip()
        if arg.startswith('const'):
            arg = arg.replace('const', '').strip()
        if ':' in arg:
            arg = arg.split(':')[0]
        match = re.match(r'^[\w\.]+\s+(\w.+)', arg)
        if match:
            arg = match.group(1)
        args[idx] = arg.strip()
    return ', '.join(args)


def convert_getter(source, start):
    """
    Process a function that is wrapped with @property decorator
    """
    current = source[start + 1].strip()
    if not current.startswith('def'):  # Base Case
        return
    new_sign = get_signature(source, start + 1)
    if new_sign:
        new_func, shift = new_sign
        new_func += ':'
        # get comments
        comments = []
        if start > 1:
            idx = start - 1
            while source[idx].lstrip().startswith('#') and idx >= 0:
                comments.append(source[idx].strip())
                idx -= 1
            comments.reverse()
        return new_func, comments, shift


def get_signature(source, start):
    """
    Get function signature and process it
    """
    match = re.search(r'c?p?def.+\(', source[start].strip())
    if not match:
        return
    start_j = match.span()[1]
    open_brackets = 1
    new_sign = match.group()

    for i in range(start, len(source)):
        line = source[i].strip()
        for j in range(start_j, len(line)):
            char = line[j]
            if char == ')':
                open_brackets -= 1
            if char == '(':
                open_brackets += 1
            new_sign += char
            if not open_brackets:
                return new_sign + ':\n', i - start
        start_j = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pyx_file', type=str, nargs='+', help='Path to a .pyx file.')
    args = parser.parse_args()
    for pyx in args.pyx_file:
        process_pyx(pyx)


if __name__ == '__main__':
    main()
