# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import argparse
import sys

import yaml
import json
import jsonschema
import logging
from pathlib import Path
from ghapi.all import GhApi
from fnmatch import fnmatch


class ComponentConfig:
    FullScope = {'build', 'test'}
    ScopeKeys = {'build', 'revalidate'}

    def __init__(self, config: dict, schema: dict, all_possible_components: set):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.all_defined_components = set(self.config.keys())  # already defined in components.yml
        # can be added to components.yml (based on labeler.yml)
        self.all_possible_components = all_possible_components.union(self.all_defined_components)
        self.validate(schema)

    def validate(self, schema: dict) -> None:
        """Validates syntax of configuration file"""
        jsonschema.validate(self.config, schema)

        for component_name, data in self.config.items():
            dependent_components = set()
            for key in self.ScopeKeys:
                scope = data.get(key)
                dependent_components = dependent_components.union(set(scope) if isinstance(scope, list) else set())

            invalid_dependents = dependent_components.difference(self.all_possible_components)
            if invalid_dependents:
                error_msg = f"dependent components of {component_name} are invalid: " \
                            f"{invalid_dependents} are not listed in components config: {self.all_possible_components}"
                raise jsonschema.exceptions.ValidationError(error_msg)

    def get_affected_components(self, changed_components_names: set) -> dict:
        """Returns changed components, their dependencies and validation scope for them"""
        affected_components = dict()

        # If some changed components were not defined in config or no changed components detected at all,
        # run full scope for everything (just in case)
        changed_not_defined_components = changed_components_names.difference(self.all_defined_components)
        if not changed_components_names or changed_not_defined_components:
            self.log.info(f"Changed components {changed_not_defined_components} are not defined in smart ci config, "
                          "run full scope")
            affected_components.update({name: self.FullScope for name in self.all_possible_components})
            return affected_components

        # Else check changed components' dependencies and add them to affected
        for name in changed_components_names:
            component_scopes = {k: v for k, v in self.config.get(name, dict()).items() if k in self.ScopeKeys}
            for key, dependents in component_scopes.items():
                if dependents == 'all':
                    dependents = self.all_possible_components
                for dep_name in dependents:
                    affected_components[dep_name] = affected_components.get(dep_name, set())
                    scope = self.FullScope if key == 'revalidate' else {key}
                    affected_components[dep_name] = affected_components[dep_name].union(scope)

            if not component_scopes:
                self.log.info(f"Changed component '{name}' doesn't have {self.ScopeKeys} keys in components config. "
                              f"Assuming that it affects everything, the whole scope will be started")
                for dep_name in self.all_possible_components:
                    affected_components[dep_name] = self.FullScope

        # If the component was explicitly changed, run full scope for it
        affected_components.update({name: self.FullScope for name in changed_components_names})
        self.log.info(f"Changed components with dependencies: {affected_components}")

        # For non-affected components that are not defined in config - run full scope
        affected_components.update({name: self.FullScope for name in self.all_possible_components
                                    if name not in self.all_defined_components})

        return affected_components

    def get_static_data(self, components_names: set, data_key: str, default: str = None) -> dict:
        """Returns requested generic static data defined for each component"""
        data = {name: self.config[name].get(data_key, default) for name in components_names}
        return data


def component_name_from_label(label: str, component_pattern: str = None) -> str:
    """Extracts component name from label"""
    component = label
    if component_pattern:
        matches = re.findall(component_pattern, label)
        component = matches[0] if matches else None
    component = component.replace(' ', '_') if component else None
    return component


def get_changed_component_names(pr, all_possible_components: set, component_pattern: str = None) -> set:
    """Returns component names changed in a given PR"""
    components = set()
    for label in pr.labels:
        component = component_name_from_label(label.name, component_pattern)
        if component:
            components.add(component)
        elif label.name in all_possible_components:
            # Allow any labels defined explicitly in labeler config as components
            # (predefined labels, such as "do not merge", are still ignored)
            components.add(label.name)

    return components


def get_changeset(gh_api, pr, target_branch, commit_sha):
    """Returns changeset either from PR or commit"""
    if pr:
        return gh_api.pulls.list_files(pr)
    if target_branch:
        target_branch_head_commit = gh_api.repos.get_branch(target_branch).commit.sha
        # In merge-queue branch all commits between head of target branch and head of current branch (commit_sha)
        # contain changes added to queue earlier to be validated together. Getting all of them + changes from
        # commit_sha below
        changed_files = gh_api.repos.compare_commits(f'{target_branch_head_commit}...{commit_sha}').get('files', [])
        return changed_files
    raise ValueError(f'Either "pr" or "target_branch" parameter must be non-empty')


def parse_args():
    parser = argparse.ArgumentParser(description='Returns product components changed in a given PR or commit')
    parser.add_argument('--pr', type=int, required=False, help='PR number. If not set, --commit is used')
    parser.add_argument('-s', '--commit-sha', required=False, help='Commit SHA. If not set, --pr is used')
    parser.add_argument('-r', '--repo', help='GitHub repository')
    parser.add_argument('-f', '--ref_name', required=False, help='GitHub ref name')
    parser.add_argument('-p', '--pattern', default=None, help='Pattern to extract component name from PR label. '
                                                              'If not set, any label is considered a component name')
    parser.add_argument('-c', '--components-config', default='.github/components.yml',
                        help='Path to config file with info about dependencies between components')
    parser.add_argument('-m', '--components-config-schema', default='.github/actions/smart-ci/components_schema.yml',
                        help='Path to the schema file for components config')
    parser.add_argument('-l', '--labeler-config', default='.github/labeler.yml',
                        help='Path to PR labeler config file')
    parser.add_argument('--skip-when-only-listed-labels-set',
                        help="Comma-separated list of labels. If PR has only these labels set, "
                             "return indicator that CI can be skipped")
    parser.add_argument('--skip-when-only-listed-files-changed',
                        help="Comma-separated list of patterns (fnmatch-style). If PR has only matching files changed, "
                             "return indicator that CI can be skipped")
    parser.add_argument('--enable_for_org', default='openvinotoolkit',
                        help='Enable running workflows for a given organization; triggers from other orgs are skipped')
    args = parser.parse_args()
    return args


def init_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')


def set_github_output(name: str, value: str, github_output_var_name: str = 'GITHUB_OUTPUT'):
    """Sets output variable for a GitHub Action"""
    logger = logging.getLogger(__name__)
    # In an environment variable "GITHUB_OUTPUT" GHA stores path to a file to write outputs to
    with open(os.environ.get(github_output_var_name), 'a+') as file:
        logger.info(f"Add {name}={value} to {github_output_var_name}")
        print(f'{name}={value}', file=file)


def main():
    init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()
    for arg, value in sorted(vars(args).items()):
        logger.info(f"Argument {arg}: {value}")

    with open(Path(args.components_config), 'r') as config:
        components_config = yaml.safe_load(config)

    owner, repository = args.repo.split('/')

    if owner != args.enable_for_org:
        logger.info(f"Running workflows is enabled only for repos in {args.enable_for_org} organization. "
                    f"The current workflow was initiated from other org: {owner}, skipping")
        set_github_output("skip_workflow", "True")
        sys.exit(0)

    gh_api = GhApi(owner=owner, repo=repository, token=os.getenv("GITHUB_TOKEN"))
    pr = gh_api.pulls.get(args.pr) if args.pr else None

    with open(Path(args.components_config_schema), 'r') as schema_file:
        schema = yaml.safe_load(schema_file)

    with open(Path(args.labeler_config), 'r') as labeler_file:
        labeler_config = yaml.safe_load(labeler_file)

    all_possible_components = set()
    for label in labeler_config.keys():
        component_name = component_name_from_label(label, args.pattern)
        all_possible_components.add(component_name if component_name else label)

    run_full_scope = False
    # For now, we don't want to apply smart ci rules for post-commits
    is_postcommit = not pr

    merge_queue_prefix = 'gh-readonly-queue/'
    is_merge_queue = args.ref_name.startswith(merge_queue_prefix)
    merge_queue_target_branch = re.findall(f'^{merge_queue_prefix}(.*)/', args.ref_name)[0] if is_merge_queue else None

    if is_merge_queue:
        logger.info(f"The run is a merge-queue run, executing full validation scope for all components, if "
                    f"not all queued changes match patterns in 'skip-when-only-listed-files-changed'")
        run_full_scope = True
    elif is_postcommit:
        logger.info(f"The run is a post-commit run, executing full validation scope for all components")
        run_full_scope = True
    else:
        no_match_files_changed = 'no-match-files' in [label.name for label in pr.labels]
        if no_match_files_changed:
            logger.info(f"There are changed files that don't match any pattern in labeler config, "
                        f"executing full validation scope for all components")
            run_full_scope = True

    # In post-commits - validate all components regardless of changeset
    # In pre-commits - validate only changed components with their dependencies
    all_defined_components = components_config.keys()
    changed_by_pr = get_changed_component_names(pr, all_possible_components, args.pattern) if pr else None
    changed_component_names = set(all_defined_components) if run_full_scope else changed_by_pr

    logger.info(f"changed_component_names: {changed_component_names}")

    cfg = ComponentConfig(components_config, schema, all_possible_components)
    affected_components = cfg.get_affected_components(changed_component_names)

    skip_workflow = False
    if is_merge_queue or (args.pr and not run_full_scope):
        if args.skip_when_only_listed_labels_set and changed_component_names:
            excepted_labels = set(args.skip_when_only_listed_labels_set.split(','))
            excepted_labels_only = changed_component_names - excepted_labels == set()
            skip_workflow = excepted_labels_only

        if not skip_workflow and args.skip_when_only_listed_files_changed:
            # To avoid spending extra API requests running step below only if necessary
            changed_files = get_changeset(gh_api, args.pr, merge_queue_target_branch, args.commit_sha)
            patterns = set(args.skip_when_only_listed_files_changed.split(','))

            matched_files_only = all(any(fnmatch(f.filename, pattern) for pattern in patterns) for f in changed_files)
            logger.debug(f"matched files only: {matched_files_only}")
            skip_workflow = matched_files_only

    if skip_workflow:
        logger.info(f"All changes are marked for skip, workflow may be skipped")
        set_github_output("skip_workflow", str(skip_workflow))

    # Syntactic sugar for easier use in GHA pipeline
    affected_components_output = {name: {s: True for s in scope} for name, scope in affected_components.items()}
    set_github_output("affected_components", json.dumps(affected_components_output))

    # Components actually changed by a pull request are marked as True (if event is PR;
    # otherwise all components considered changed).
    changed_components_output = {name: True if not pr or name in changed_by_pr else False
                                 for name in all_possible_components}
    set_github_output("changed_components", json.dumps(changed_components_output))


if __name__ == '__main__':
    main()
