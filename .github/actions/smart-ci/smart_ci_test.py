# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import unittest
from smart_ci import ComponentConfig

log = logging.getLogger()
log.level = logging.DEBUG


def log_handler(func):
    def wrapper(*args, **kwargs):
        stream_handler = logging.StreamHandler(sys.stdout)
        log.addHandler(stream_handler)
        result = func(*args, **kwargs)
        log.removeHandler(stream_handler)
        return result
    return wrapper


class TestComponentConfig(unittest.TestCase):
    def setUp(self):
        self.all_possible_components = {'comp1', 'comp2', 'comp3', 'comp4'}
        ComponentConfig.ScopeKeys = {'build', 'revalidate', '_scope_1', '_scope_2', '_scope_3'}

    @log_handler
    def validate(self, config_data: dict, changed_components: set, expected_result: dict):
        log.info(f"{self._testMethodName}:")
        config = ComponentConfig(config_data, {}, self.all_possible_components)
        result = config.get_affected_components(changed_components)
        self.assertEqual(expected_result, result)

    def test_no_changed_components(self):
        config_data = {
            'comp1': {'build': {}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = set()
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': ComponentConfig.FullScope,
            'comp3': ComponentConfig.FullScope,
            'comp4': ComponentConfig.FullScope,
        }
        self.validate(config_data, changed_components, expected_result)

    def test_all_components_changed(self):
        config_data = {
            'comp1': {'build': {}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1', 'comp2', 'comp3', 'comp4'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': ComponentConfig.FullScope,
            'comp3': ComponentConfig.FullScope,
            'comp4': ComponentConfig.FullScope,
        }
        self.validate(config_data, changed_components, expected_result)

    def test_changed_component_not_defined(self):
        config_data = {
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': ComponentConfig.FullScope,
            'comp3': ComponentConfig.FullScope,
            'comp4': ComponentConfig.FullScope,
        }
        self.validate(config_data, changed_components, expected_result)

    def test_component_changed_no_scope_keys(self):
        config_data = {
            'comp1': {},
            'comp2': {},
            'comp3': {},
            'comp4': {},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': ComponentConfig.FullScope,
            'comp3': ComponentConfig.FullScope,
            'comp4': ComponentConfig.FullScope,
        }
        self.validate(config_data, changed_components, expected_result)

    def test_one_component_changed_dependents_empty(self):
        config_data = {
            'comp1': {'build': {}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
        }
        self.validate(config_data, changed_components, expected_result)

    def test_not_changed_dependent_component(self):
        config_data = {
            'comp1': {'build': {'comp2'}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': {'build'}
        }
        self.validate(config_data, changed_components, expected_result)

    def test_changed_dependent_component(self):
        config_data = {
            'comp1': {'build': {'comp2'}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1', 'comp2'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': ComponentConfig.FullScope
        }
        self.validate(config_data, changed_components, expected_result)

    def test_dependent_component_multiple_parents(self):
        config_data = {
            'comp1': {'_scope_1': {'comp2'}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, '_scope_2': {'comp2'}, '_scope_3': {'comp2'}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1', 'comp3'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': {'_scope_1', '_scope_2', '_scope_3'},
            'comp3': ComponentConfig.FullScope
        }
        self.validate(config_data, changed_components, expected_result)

    def test_dependent_component_empty_scopes(self):
        config_data = {
            'comp1': {'build': {}, 'revalidate': {'comp2'}},
            'comp2': {},
            'comp3': {},
            'comp4': {},
        }
        changed_components = {'comp1', 'comp3'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': ComponentConfig.FullScope,
            'comp3': ComponentConfig.FullScope,
            'comp4': ComponentConfig.FullScope
        }
        self.validate(config_data, changed_components, expected_result)

    def test_changed_component_empty_dependencies(self):
        config_data = {
            'comp1': {'build': {}, 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
        }
        self.validate(config_data, changed_components, expected_result)

    def test_multiple_dependents(self):
        config_data = {
            'comp1': {'build': {'comp2'}, 'revalidate': {'comp3'}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {'comp4'}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': {'build'},
            'comp3': ComponentConfig.FullScope,
            # We don't consider dependencies of dependencies affected, so comp4 is not expected here
        }
        self.validate(config_data, changed_components, expected_result)

    def test_all_as_dependents(self):
        config_data = {
            'comp1': {'build': 'all', 'revalidate': {}},
            'comp2': {'build': {}, 'revalidate': {}},
            'comp3': {'build': {}, 'revalidate': {}},
            'comp4': {'build': {}, 'revalidate': {}},
        }
        changed_components = {'comp1'}
        expected_result = {
            'comp1': ComponentConfig.FullScope,
            'comp2': {'build'},
            'comp3': {'build'},
            'comp4': {'build'},
        }
        self.validate(config_data, changed_components, expected_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
