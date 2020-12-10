import argparse
import os
from lxml import etree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openvino', type=str, required=True, default=None, help='openvino_docs.xml')
    args = parser.parse_args()
    result = build_layout(args.openvino)
    with open(args.openvino, 'wb') as f:
        f.write(result)


def build_layout(openvino):
    ns = {"xi": "http://www.w3.org/2001/XInclude"}
    root = etree.parse(openvino)
    root.xinclude()
    return etree.tostring(root, pretty_print=True)


if __name__ == '__main__':
    main()
