import logging as log
import sys
import os
import csv
import pytest
import re
from conftest import shell

log.basicConfig(format="[ %(levelname)s ]  %(msg)s", stream=sys.stdout, level=log.INFO)

pytest.operation_dictionary = {}
pytest.avaliable_plugins = []


def save_coverage_to_csv(csv_path, header):
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(i for i in header)
        i = 1
        for key in sorted(pytest.operation_dictionary):
            line = [i, key]
            for plugin in pytest.avaliable_plugins:
                if not plugin in pytest.operation_dictionary[key]:
                    line.append('0/0')
                else:
                    line.append('/'.join(str(x) for x in pytest.operation_dictionary[key][plugin]))
            csv_writer.writerow(line)
            i += 1


def get_color(value):
    if '/' in value:
        passed, total = [int(x.strip()) for x in value.split('/')]
        if passed == total and total != 0:
            return "#d1ffd3"
        elif passed == total and total == 0:
            return "#dadada"
        else:
            return "#ffdbdb"
    else:
        return "white"


def csv_to_html_table(csv_path, html_path, headers=None, delimiter=","):
    with open(csv_path) as f:
        content = f.readlines()

    # reading file content into list
    rows = [x.strip() for x in content]
    table = "<!DOCTYPE html><html><head><title>Opset1 operations results</title></head><body><table border=1>"

    # creating HTML header row if header is provided
    if headers is not None:
        table += "<tr>"
        table += "".join(["<th>" + cell + "</th>" for cell in headers])
        table += "</tr>"
    else:
        table += "<tr>"
        table += "".join(["<th>" + cell + "</th>" for cell in rows[0].split(delimiter)])
        table += "</tr>"
        rows = rows[1:]

    # Converting csv to html row by row
    for row in rows:
        table += "<tr>" + "".join(["<td style=background-color:%s>" % (get_color(cell)) + cell + "</td>"
                                   for cell in row.split(delimiter)]) + "</tr>" + "\n"
    table += "</table></body></html><br>"

    # Saving html file
    fh = open(html_path, "w")
    fh.write(table)
    fh.close()


def setup_module():
    try:
        os.environ.get('PATH_TO_EXE')
    except KeyError:
        raise ImportError('PATH_TO_EXE is upsent in your environment variables. '
                          'Please, do "export PATH_TO_EXE=<path to unit-test>')


def teardown_module():
    """
    Creating CSV file at the end of test with nGraph nodes coverage
    :return:
    """
    csv_path = "nodes_coverage.csv"
    header = ["#", "Operation"] + [p + " passed / total" for p in pytest.avaliable_plugins]
    save_coverage_to_csv(csv_path=csv_path, header=header)

    # Convert csv file to html for better visualization
    html_path = "nodes_coverage.html"
    csv_to_html_table(csv_path=csv_path, html_path=html_path, delimiter="|")


def test(gtest_filter):
    executable = os.path.join(os.environ.get('PATH_TO_EXE'), "unit-test")
    cmd_line = executable + ' --gtest_filter=' + gtest_filter
    retcode, stdout = shell(cmd=cmd_line)

    # Parsing output of single test
    stdout = stdout.split('\n')
    nodes_list = []
    for line in stdout:
        if 'UNSUPPORTED OPS DETECTED!' in line:
            pytest.skip('Skip from pytest because unit-test send error UNSUPPORTED OPS DETECTED!')
        elif 'Nodes in test:' in line:
            nodes_list = list(set(line.replace('Nodes in test:', '').strip().split(' ')))

    if not nodes_list:
        pytest.skip('Skip from pytest because inside test no one ngraph function created')

    # Added one more loop, because condition below must be executed only if some nodes_list found
    # (it means that test includes opset1 operations)
    for line in stdout:
        if re.match('.*1 test from\s([A-Z]+)', line):
            matches = re.search(r'.*1 test from\s([A-Z]+)', line)
            plugin = matches.group(1)
            if plugin not in pytest.avaliable_plugins:
                pytest.avaliable_plugins.append(plugin)

    # Filling dictionary with operation coverage
    # How many time one operation is tested
    for n in nodes_list:
        if plugin in pytest.operation_dictionary[n]:
            numerator, denominator = pytest.operation_dictionary[n][plugin]
            pytest.operation_dictionary[n][plugin] = (numerator if retcode != 0 else numerator + 1,
                                                      denominator + 1)
        else:
            pytest.operation_dictionary[n][plugin] = (0, 1) if retcode != 0 else (1, 1)

    # This check is at the end, because with 99% it will return 0 or 1 (when function check of test failed)
    # Because the same cmd line executed by pytest_generate_tests with --gtest_list_tests.
    # So, most of the issue cached there.
    assert retcode == 0, "unit-test execution failed. Gtest failed. Return code: {}".format(retcode)


if __name__ == '__main__':
    log.warning("Please run {} by pytest like so:\npytest {} --gtest_filter=<attributes for gtest_filter>")
