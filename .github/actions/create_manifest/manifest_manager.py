# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict, List, Union, Iterator, Any


class ManifestException(Exception):
    """Base Manifest file manager exception"""


class ManifestDoesNotExist(ManifestException):
    """ManifestDoesNotExist Manifest file manager exception"""


class ManifestSavingError(ManifestException):
    """ManifestSavingError Manifest file manager exception"""


class WrongComponentFormatError(ManifestException):
    """WrongComponentFormatError Manifest file manager exception"""


class WrongRepositoryFormatError(ManifestException):
    """WrongRepositoryFormatError Manifest file manager exception"""


class Manifest:
    """Manifest wrapper"""

    default_manifest_name = "manifest.yml"

    def __init__(self, manifest_path: Optional[str] = None):
        """
        :param manifest_path: Path to a manifest file
        """
        self._manifest_file = Path(manifest_path or self.default_manifest_name)
        if self._manifest_file.is_dir():
            self._manifest_file = self._manifest_file / self.default_manifest_name

        self._manifest_version = "1.0"
        self._components: Dict[str, Component] = {}

        if manifest_path is not None:
            self._prepare_manifest()

    def __repr__(self) -> str:
        return str(self._manifest_file)

    def _prepare_manifest(self) -> None:
        """Read manifest file and convert its data to objects"""
        if not self._manifest_file.is_file():
            raise ManifestDoesNotExist(f'Cannot find manifest "{self._manifest_file}"')

        with self._manifest_file.open("r") as manifest:
            manifest_info = yaml.safe_load(manifest)

        if not isinstance(manifest_info, dict):
            raise ManifestDoesNotExist(f'Incorrect manifest "{self._manifest_file}"')

        self._manifest_version = manifest_info.get("manifest_version", self._manifest_version)

        for name, info in manifest_info["components"].items():
            self._components[name] = Component.from_dict({
                "name": name,
                "version": info["version"],
                "repository": info["repository"],
                "product_type": info["product_type"],
                "target_arch": info["target_arch"],
                "build_type": info["build_type"],
                "build_event": info["build_event"],
                "custom_params": info.get("custom_params")
            })

    @property
    def version(self) -> str:
        return self._manifest_version

    @property
    def components(self) -> List[Component]:
        return list(self._components.values())

    def get_component(self, component_name: str) -> Optional[Component]:
        return self._components.get(component_name)

    def add_component(self, component: Component, replace: bool = False) -> bool:
        if not replace and component.name in self._components:
            return False
        self._components[component.name] = component
        return True

    def delete_component(self, component_name: str) -> bool:
        return self._components.pop(component_name, None) is not None

    def save_manifest(self, save_to: Union[str, Path]) -> None:
        class YamlDumper(yaml.SafeDumper):
            """Formatting PyYAML dump() output"""

            def write_line_break(self, data=None):
                super().write_line_break(data)
                if len(self.indents) in {1, 2, 4}:
                    super().write_line_break()

        path_to_save = Path(save_to)
        if path_to_save.is_dir():
            path_to_save = path_to_save / self.default_manifest_name
        else:
            path_to_save.parent.mkdir(parents=True, exist_ok=True)

        manifest_data = {"components": {}, "manifest_version": self._manifest_version}
        for comp_name, comp_data in self._components.items():
            comp = dict(comp_data)
            manifest_data["components"][comp_name] = {
                "version": comp["version"],
                "product_type": comp["product_type"],
                "target_arch": comp["target_arch"],
                "build_type": comp["build_type"],
                "build_event": comp["build_event"],
                "trigger_repo_name": comp["trigger_repo_name"],
                "custom_params": comp["custom_params"],
                "repository": comp["repositories"],
            }

        try:
            with path_to_save.open("w") as manifest:
                yaml.dump(manifest_data, stream=manifest, Dumper=YamlDumper, default_flow_style=False, sort_keys=False)
        except Exception as ex:
            raise ManifestSavingError(ex) from ex

    def as_dict(self) -> Dict[str, Union[str, Dict]]:
        """Return manifest as dictionary"""
        if not self._manifest_file.is_file():
            raise ManifestDoesNotExist(f'Cannot find manifest "{self._manifest_file}"')

        with self._manifest_file.open("r") as manifest:
            manifest_dict = yaml.safe_load(manifest)

        if not isinstance(manifest_dict, dict):
            raise ManifestDoesNotExist(f'Incorrect manifest "{self._manifest_file}"')

        return manifest_dict


class Repository:
    def __init__(self, **kwargs) -> None:
        self._state: dict = {
            "name": None,
            "url": None,
            "branch": None,
            "revision": None,
            "commit_id": None,
            "commit_time": None,
            "target_branch": None,
            "target_revision": None,
            "target_commit_id": None,
            "merge_target": False,
            "revert_time": None,
            "trigger": False,
            "default_branch": None,
            "type": "git",
        }
        for arg_name, arg_value in kwargs.items():
            if arg_name in self._state:
                self._state[arg_name] = arg_value

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self._state:
            return self._state.get(attr_name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")

    def __iter__(self) -> Iterator:
        for name in self._state:
            yield name, self._state.get(name)

    def get_git_repo_state(self) -> dict:
        state = deepcopy(self._state)
        state.pop("revision")
        state.pop("target_revision")
        state.pop("commit_time")
        state.pop("type")
        state["commit_id"] = self._state["revision"]
        state["target_commit_id"] = self._state["target_revision"]
        return state


class Component:
    def __init__(
        self,
        name: str,
        version: str,
        repositories: list,
        product_type: str,
        target_arch: str,
        build_type: str,
        build_event: str,
        custom_params: Optional[dict] = None
    ):
        """
        Initialize the product component.

        :param name: Name of component
        :param version: Version of component
        :param repositories: List of repositories
        :param product_type: Unique key to describe a product type (can include OS, arch, build variant, etc)
        :param target_arch: Target architecture
        :param build_type: Type of build (release, debug)
        :param build_event: Build event (pre_commit, commit)
        :param custom_params: Custom parameters (optional)
        """
        self._name = name
        self._version = version
        self._repositories = {}
        self._product_type = product_type
        self._target_arch = target_arch
        self._build_type = build_type
        self._build_event = build_event
        self._custom_params = custom_params if custom_params is not None else {}
        self._trigger_repo_name = None

        self._prepare_repositories(repositories)

    def __iter__(self) -> Iterator:
        yield "name", self._name
        yield "version", self._version
        yield "product_type", self._product_type
        yield "target_arch", self._target_arch
        yield "build_type", self._build_type
        yield "build_event", self._build_event
        yield "trigger_repo_name", self._trigger_repo_name
        yield "custom_params", self._custom_params
        yield "repositories", [dict(repo) for repo in self._repositories.values()]

    def _prepare_repositories(self, repositories: list) -> None:
        for repo in repositories:
            repo_name, repo_obj = self._parse_repository(repo)
            self._repositories[repo_name] = repo_obj

            if repo_obj.trigger:
                if self._trigger_repo_name:
                    raise WrongRepositoryFormatError(
                        f"Found trigger repo duplicates: {self._trigger_repo_name}, {repo_name}"
                    )
                self._trigger_repo_name = repo_name

    @staticmethod
    def _parse_repository(repo: Union[dict, Repository]) -> tuple[str, Repository]:
        if isinstance(repo, dict):
            repo_name = repo["name"]
            repo_obj = Repository(**repo)
        elif isinstance(repo, Repository):
            repo_name = repo.name
            repo_obj = repo
        return repo_name, repo_obj

    @staticmethod
    def from_dict(comp_data: dict) -> Component:
        """
        Convert a dictionary to a Component object.

        :param comp_data: Component data dictionary
        :return: Component object
        """
        try:
            return Component(
                comp_data["name"],
                comp_data["version"],
                comp_data["repository"],
                comp_data["product_type"],
                comp_data["target_arch"],
                comp_data["build_type"],
                comp_data["build_event"],
                comp_data.get("custom_params"),
            )
        except Exception as ex:
            raise WrongComponentFormatError(ex) from ex

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def product_type(self) -> str:
        return self._product_type

    @property
    def target_arch(self) -> str:
        return self._target_arch

    @property
    def build_type(self) -> str:
        return self._build_type

    @property
    def build_event(self) -> str:
        return self._build_event

    @property
    def repositories(self) -> List[Repository]:
        return list(self._repositories.values())

    @property
    def trigger_repo_name(self) -> Optional[str]:
        return self._trigger_repo_name

    @property
    def trigger_repository(self) -> Optional[Repository]:
        return next((repo for repo in self._repositories.values() if repo.trigger), None)

    def get_repository(self, repository_name: str) -> Optional[Repository]:
        return self._repositories.get(repository_name)

    def add_repository(self, repository: Repository, replace: bool = False) -> bool:
        if not replace and repository.name in self._repositories:
            return False
        self._repositories[repository.name] = repository
        return True

    def delete_repository(self, repository_name: str) -> bool:
        return self._repositories.pop(repository_name, None) is not None

    def get_custom_param(self, name: str) -> Optional[Any]:
        return self._custom_params.get(name)

    def add_custom_param(self, name: str, value: Any) -> None:
        self._custom_params[name] = value

    def delete_custom_param(self, name: str) -> bool:
        return self._custom_params.pop(name, None) is not None
