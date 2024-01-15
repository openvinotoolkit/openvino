"""Main entry-point to get reference results for Dynamism: first time inference test
Default run:
$ python get_reference_res_from_html.py

Options[*]:
--html         Path to HTML
--ref_yml_path    Path to yml reference file

"""

import argparse
import codecs
import re

import yaml


def get_tests_results(html_path: str):
    html_file = codecs.open(html_path, 'r')
    html_info = html_file.read()
    pattern = r'<td class="col-result">([a-zA-Z]+)</td>[.\s]+<td class="col-name">[\w:.]+\[(.*)\]</td>'
    pattern_short_log = r'<div class="log">([\s\S]+)<br/>-+ Captured stdout'
    pattern_log = r'-+<br/>([\s\S]+)-+ Captured log call'
    res = []
    for item in html_info.split('<tbody'):
        tmp_list = re.findall(pattern, item)
        short_log = re.findall(pattern_short_log, item)
        log = re.findall(pattern_log, item)
        if tmp_list:
            res.append([tmp_list[0][0], tmp_list[0][1], short_log[0] if short_log else '', log[0] if log else ''])

    return {key: [value, short_log, log] for value, key, short_log, log in res}


def get_reference_values(results: dict):
    def _refs_refactor(ref_str: str):
        ref_collector = {}
        refactor_metrics = list(filter(None, ref_str.replace('[ INFO ] ', '').replace(' %', '').split('\n')))
        for metric in refactor_metrics:
            metric_values = metric.split(' | ')
            ref_collector.update({metric_values[0]: metric_values[1]})
        return ref_collector

    ref_res = {}
    ref_pattern = r'as a percentage:[.\s]+([\s\S]+)\[ INFO \] Difference between current'
    for test_key, values in results.items():
        test_ref_name = re.sub(r'_api(.*)_precision_\w+\d{2}', '', test_key).replace('__', '_')
        ref_values = re.findall(ref_pattern, values[-1])
        if not ref_values:
            continue
        ref_values = _refs_refactor(ref_values[0])
        ref_res.update({test_ref_name: ref_values})
    return ref_res


def main(args):
    test_results = get_tests_results(args.html)
    reference_values = get_reference_values(test_results)

    with open(args.ref_yml_path, 'r') as config_file:
        ref_content = yaml.safe_load(config_file)

    for ref_name, ref_values in reference_values.items():
        if ref_name in ref_content:
            ref_content.update({ref_name: ref_values})

    with open(args.ref_yml_path, 'w') as config_file:
        yaml.safe_dump(ref_content, config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--html', help='Path to HTML', type=str)
    parser.add_argument('--ref_yml_path', help='Path to yml reference file', type=str)
    args = parser.parse_args()
    main(args)
