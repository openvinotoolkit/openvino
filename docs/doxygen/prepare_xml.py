import re
import logging
import argparse
from pathlib import Path
from xml.sax import saxutils


def prepare_xml(xml_dir: Path):
    pattern = r'\<sphinxdirective\>(.+?)\<\/sphinxdirective>'
    xml_files = xml_dir.glob('*.xml')

    for xml_file in xml_files:
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                contents = f.read()
            matches = re.findall(pattern, contents, flags=re.DOTALL)
            if matches:
                for match in matches:
                    contents = contents.replace(match, saxutils.escape(match))
                with open(xml_file, 'w', encoding='utf-8') as f:
                    f.write(contents)
        except UnicodeDecodeError as err:
            logging.warning('{}:{}'.format(xml_file, err))


def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_dir', type=Path, help='Path to the folder containing xml files.')
    args = parser.parse_args()
    xml_dir = args.xml_dir
    prepare_xml(xml_dir)


if __name__ == '__main__':
    main()
