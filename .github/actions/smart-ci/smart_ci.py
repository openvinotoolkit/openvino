import os
import re
import argparse
import yaml
import json
import jsonschema
import logging
from pathlib import Path
from github import Github, PullRequest


class ComponentConfig:
    FullScope = ['build', 'test']

    def __init__(self, config: dict, schema: dict = None):
        self.config = config
        self.validate(schema)

    def validate(self, schema: dict) -> None:
        """Validates syntax of configuration file"""
        jsonschema.validate(self.config, schema)
        all_components = set(self.config.keys())

        for component_name, data in self.config.items():
            dependent_components = set(data.get('dependent_components', dict()).keys()) if data else set()
            invalid_dependents = dependent_components.difference(all_components)
            if invalid_dependents:
                error_msg = f"dependent_components of {component_name} are invalid: " \
                            f"{invalid_dependents} are not listed in {all_components}"
                raise jsonschema.exceptions.ValidationError(error_msg)

    def get_affected_components(self, changed_components_names: set) -> dict:
        """Returns changed components, their dependencies and validation scope for them"""
        affected_components = dict()
        for name in changed_components_names:
            dependent_components = self.config[name].get('dependent_components', dict())
            affected_components.update({dep_name: scope if scope else self.FullScope
                                        for dep_name, scope in dependent_components.items()})
        # We want to run the full scope if the component was explicitly changed
        affected_components.update({name: self.FullScope for name in changed_components_names})
        return affected_components

    def get_static_data(self, components_names: set, data_key: str, default: str = None) -> dict:
        """Returns generic static data defined for each component, if any"""
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


def get_changed_component_names(pr: PullRequest, component_pattern: str = None) -> set:
    """Returns component names changed in a given PR"""
    components = set()
    for label in pr.labels:
        component = component_name_from_label(label.name, component_pattern)
        if component:
            components.add(component)
    return components


def parse_args():
    parser = argparse.ArgumentParser(description='Returns product components changed in a given PR or commit')
    parser.add_argument('--pr', type=int, required=False, help='PR number. If not set, --commit is used')
    parser.add_argument('-s', '--commit-sha', required=False, help='Commit SHA. If not set, --pr is used')
    parser.add_argument('-r', '--repo', help='GitHub repository')
    parser.add_argument('-p', '--pattern', default=None, help='Pattern to extract component name from PR label. '
                                                              'If not set, any label is considered a component name')
    parser.add_argument('-c', '--components-config', default='.github/components.yml',
                        help='Path to config file with info about dependencies between components')
    parser.add_argument('-m', '--components-config-schema', default='.github/actions/smart-ci/components_schema.yml',
                        help='Path to the schema file for components config')
    args = parser.parse_args()
    return args


def init_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
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

    gh_api = Github(os.getenv("GITHUB_TOKEN"))
    repository = gh_api.get_repo(args.repo)
    pr = repository.get_pull(args.pr) if args.pr else None  # get_pr_by_commit(repository, args.commit_sha)

    # For now, we don't want to apply smart ci rules for post-commits
    is_postcommit = not pr
    if is_postcommit:
        logger.info(f"The run is a post-commit run, executing full validation scope for all components")

    # In post-commits - validate all components regardless of changeset
    # In pre-commits - validate only changed components with their dependencies
    all_components = components_config.keys()
    changed_component_names = all_components if is_postcommit else get_changed_component_names(pr, args.pattern)
    logger.info(f"changed_component_names: {changed_component_names}")

    schema_path = Path(args.components_config_schema)
    with open(schema_path, 'r') as schema_file:
        schema = yaml.safe_load(schema_file)

    cfg = ComponentConfig(components_config, schema)
    affected_components = cfg.get_affected_components(changed_component_names)

    # Syntactic sugar for easier use in GHA pipeline
    all_components_output = {component: True for component in all_components}
    affected_components_output = {name: {s: True for s in scope} for name, scope in affected_components.items()}

    # We need to know which components were defined in a config to be able to apply smart ci rules only to them.
    # For those not defined we want to run everything by default
    set_github_output("all_components", json.dumps(all_components_output))
    set_github_output("affected_components", json.dumps(affected_components_output))


if __name__ == '__main__':
    main()
