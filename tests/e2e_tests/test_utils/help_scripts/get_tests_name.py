"""Main entry-point to get names of E2E OSS tests containing skip MO args

Default run:
$ pytest get_tests_name.py --modules=<path_to_tests> --skip_mo_args=<list_of_mo_args>

Options[*]:
--modules         Paths to tests
--skip_mo_args    Path to test config

[*] For more information see conftest.py
"""

import os
import pytest
pytest_plugins = ('e2e_tests.common.plugins.e2e_test.conftest',)


@pytest.fixture(scope="session")
def list_tests(modules, skip_mo_args):
    """
    Fixture to storing test names and saving names to result_file
    :param modules:
    :return:
    """
    seporator = ' or ' if skip_mo_args else '\n  '
    path_to_result = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.txt')
    result_file = open(path_to_result, 'w')
    result_file.write(f'{modules}\n')
    list_of_tests = []
    yield list_of_tests
    result_file.write(seporator.join(list_of_tests))
    result_file.close()


def test_run(instance, skip_mo_args, list_tests):
    """
    :param instance: test instance
    :param skip_mo_args: line with comma separated args that will be deleted from MO cmd line
    :return:
    """
    if skip_mo_args:
        """
        Getting test names which using specific keys in MO
        """
        mo_args = set(instance.ie_pipeline['get_ir']['get_ovc_model']['additional_args'].keys())
        skip_mo_args = skip_mo_args.split(',')
        if mo_args.intersection(skip_mo_args):
            list_tests.append(instance.__class__.__name__)
    else:
        """
        Getting all names of test 
        """
        list_tests.append(instance.__class__.__name__)
    return
