from pathlib import Path

from conan.internal.graph.graph import CONTEXT_BUILD


class ConanFileInterface:
    """ this is just a protective wrapper to give consumers
    a limited view of conanfile dependencies, "read" only,
    and only to some attributes, not methods
    """
    def __str__(self):
        return str(self._conanfile)

    def __init__(self, conanfile):
        self._conanfile = conanfile

    def __eq__(self, other):
        """
        The conanfile is a different entity per node, and conanfile equality is identity
        :type other: ConanFileInterface
        """
        return self._conanfile == other._conanfile

    def __hash__(self):
        return hash(self._conanfile)

    @property
    def options(self):
        return self._conanfile.options

    @property
    def recipe_folder(self):
        return self._conanfile.recipe_folder

    @property
    def recipe_metadata_folder(self):
        return self._conanfile.recipe_metadata_folder

    @property
    def package_folder(self):
        return self._conanfile.package_folder

    @property
    def immutable_package_folder(self):
        return self._conanfile.immutable_package_folder

    @property
    def package_metadata_folder(self):
        return self._conanfile.package_metadata_folder

    @property
    def package_path(self) -> Path:
        assert self.package_folder is not None, "`package_folder` is `None`"
        return Path(self.package_folder)

    @property
    def ref(self):
        return self._conanfile.ref

    @property
    def pref(self):
        return self._conanfile.pref

    @property
    def buildenv_info(self):
        return self._conanfile.buildenv_info

    @property
    def runenv_info(self):
        return self._conanfile.runenv_info

    @property
    def cpp_info(self):
        return self._conanfile.cpp_info

    @property
    def settings(self):
        return self._conanfile.settings

    @property
    def settings_build(self):
        return self._conanfile.settings_build

    @property
    def context(self):
        return self._conanfile.context

    @property
    def conf_info(self):
        return self._conanfile.conf_info

    @property
    def generator_info(self):
        return self._conanfile.generator_info

    @property
    def dependencies(self):
        return self._conanfile.dependencies

    @property
    def folders(self):
        return self._conanfile.folders

    @property
    def is_build_context(self):
        return self._conanfile.context == CONTEXT_BUILD

    @property
    def package_type(self):
        return self._conanfile.package_type

    @property
    def languages(self):
        return self._conanfile.languages

    @property
    def info(self):
        return self._conanfile.info

    def set_deploy_folder(self, deploy_folder):
        self._conanfile.set_deploy_folder(deploy_folder)

    @property
    def conan_data(self):
        return self._conanfile.conan_data

    @property
    def license(self):
        return self._conanfile.license

    @property
    def description(self):
        return self._conanfile.description

    @property
    def homepage(self):
        return self._conanfile.homepage

    @property
    def url(self):
        return self._conanfile.url

    @property
    def extension_properties(self):
        return getattr(self._conanfile, "extension_properties", {})

    @property
    def recipe(self) -> str:
        # IMPORTANT: this should be used only for "informational" purposes, see GH#18996.
        return self._conanfile._conan_node.recipe

    @property
    def conf(self):
        return self._conanfile.conf
