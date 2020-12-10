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
    parser.add_argument('--exclude-links', nargs='+', type=str, required=False, default=[], help='Markdown links to be excluded')
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


def parse(log, ignore_list, strip, exclude_links):
    found_errors = []
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
    parse(args.log,  args.ignore_list, args.strip, args.exclude_links)


if __name__ == '__main__':
    main()
