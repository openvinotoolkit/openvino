# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smart_ci import ComponentConfig, main

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


class TestAutomationOnlyChanges(unittest.TestCase):
    # Non-Markdown paths from the rerunner proposal PR, alongside agentic sources and documentation.
    automation_files = [
        '.claude/skills/ov-agentic-workflows/SKILL.md',
        '.github/aw/actions-lock.json',
        '.github/ci-doctor-mq/schemas/pattern.schema.json',
        '.github/scripts/agentic-workflows/propose_rerun_pattern.py',
        '.github/scripts/workflow_rerun/argument_parser.py',
        '.github/scripts/workflow_rerun/errors_to_look_for.json',
        '.github/scripts/workflow_rerun/log_analyzer.py',
        '.github/scripts/workflow_rerun/rerunner.py',
        '.github/scripts/workflow_rerun/tests/data/empty_errors.json',
        '.github/scripts/workflow_rerun/tests/data/patterns/aaaa1111bbbb2222.json',
        '.github/scripts/workflow_rerun/tests/data/patterns/cccc3333dddd4444.json',
        '.github/scripts/workflow_rerun/tests/integration_test.py',
        '.github/scripts/workflow_rerun/tests/log_analyzer_test.py',
        '.github/workflows/ci-doctor-mq.md',
        '.github/workflows/ci-doctor-mq.lock.yml',
        '.github/workflows/ci-doctor-post-commit.lock.yml',
        '.github/workflows/ci-doctor-remediation.lock.yml',
        '.github/workflows/ci-doctor.lock.yml',
        '.github/workflows/shared/agentic-workflows/propose-rerun-pattern.md',
        '.github/workflows/workflow_rerunner.yml',
        '.github/workflows/agentic_workflows_check.yml',
        'docs/dev/ci/github_actions/agentic_workflows.md',
    ]

    def run_smart_ci(self, files, event='pull_request', labels=('category: CI', 'github_actions', 'category: docs'),
                     skip_patterns='*.md,*.rst,*.png,*.jpg,*.svg'):
        root = Path(__file__).resolve().parents[3]
        argv = [
            'smart_ci.py', '-r', 'openvinotoolkit/openvino', '-p', 'category: (.*)',
            '-c', str(root / '.github/components.yml'),
            '-m', str(root / '.github/actions/smart-ci/components_schema.yml'),
            '-l', str(root / '.github/labeler.yml'),
            '--skip-when-only-listed-files-changed', skip_patterns,
            '-f', 'gh-readonly-queue/master/pr-39282-abc' if event == 'merge_group' else 'master',
        ]
        if event == 'pull_request':
            argv.extend(['--pr', '39282'])
        outputs = {}
        pr = SimpleNamespace(labels=[SimpleNamespace(name=name) for name in labels])
        changed_files = [SimpleNamespace(filename=name) for name in files]
        with patch.object(sys, 'argv', argv), \
                patch('smart_ci.GhApi') as api, \
                patch('smart_ci.get_changeset', return_value=changed_files), \
                patch('smart_ci.set_github_output', side_effect=outputs.__setitem__):
            api.return_value.pulls.get.return_value = pr
            main()
        return outputs.get('skip_workflow') == 'True'

    def test_automation_only_pr_and_merge_queue_skip(self):
        for event in ('pull_request', 'merge_group'):
            with self.subTest(event=event):
                self.assertTrue(self.run_smart_ci(self.automation_files, event))

    def test_no_match_label_does_not_prevent_file_based_skip(self):
        self.assertTrue(self.run_smart_ci(self.automation_files, labels=('no-match-files',)))

    def test_each_automation_path_skips_without_workflow_specific_patterns(self):
        for path in self.automation_files[:-1]:
            with self.subTest(path=path):
                self.assertTrue(self.run_smart_ci([path], skip_patterns=''))

    def test_mixed_product_or_ci_changes_do_not_skip(self):
        for path in (
            'src/core/src/model.cpp',
            '.github/workflows/ubuntu_22.yml',
            '.github/workflows/job_cxx_unit_tests.yml',
            '.github/actions/smart-ci/smart_ci.py',
            '.github/labeler.yml',
            'unclassified-file',
        ):
            for event in ('pull_request', 'merge_group'):
                with self.subTest(path=path, event=event):
                    self.assertFalse(self.run_smart_ci(self.automation_files + [path], event))

    def test_empty_changeset_does_not_skip(self):
        self.assertFalse(self.run_smart_ci([]))

    def test_postcommit_does_not_skip(self):
        self.assertFalse(self.run_smart_ci(self.automation_files, 'push'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
