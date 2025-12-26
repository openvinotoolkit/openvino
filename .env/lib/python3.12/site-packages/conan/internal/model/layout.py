import os

from conan.internal.model.cpp_info import CppInfo
from conan.internal.model.conf import Conf


class Infos(object):

    def __init__(self):
        self.source = CppInfo()
        self.build = CppInfo()
        self.package = CppInfo(set_defaults=True)


class PartialLayout(object):
    def __init__(self):
        from conan.tools.env import Environment
        self.buildenv_info = Environment()
        self.runenv_info = Environment()
        self.conf_info = Conf()

    def set_relative_base_folder(self, folder):
        self.buildenv_info.set_relative_base_folder(folder)
        self.runenv_info.set_relative_base_folder(folder)
        self.conf_info.set_relative_base_folder(folder)


class Layouts(object):
    def __init__(self):
        self.source = PartialLayout()
        self.build = PartialLayout()
        self.package = PartialLayout()


class Folders:

    def __init__(self):
        self._base_source = None
        self._base_build = None
        self._base_package = None
        self._base_generators = None

        self._base_export = None
        self._base_export_sources = None

        self._base_recipe_metadata = None
        self._base_pkg_metadata = None
        self._immutable_package_folder = None

        self.source = ""
        self.build = ""
        self.package = ""
        self.generators = ""
        # Relative location of the project root, if the conanfile is not in that project root, but
        # in a subfolder: e.g: If the conanfile is in a subfolder then self.root = ".."
        self.root = None
        # The relative location with respect to the project root of the subproject containing the
        # conanfile.py, that makes most of the output folders defined in layouts (cmake_layout, etc)
        # start from the subproject again
        self.subproject = None
        self.build_folder_vars = None

    def set_base_folders(self, conanfile_folder, output_folder):
        """ this methods can be used for defining all the base folders in the
        local flow (conan install, source, build), where only the current conanfile location
        and the potential --output-folder user argument are the folders to take into account
        If the "layout()" method defines a self.folders.root = "xxx" it will be used to compute
        the base folder

        @param conanfile_folder: the location where the current consumer conanfile is
        @param output_folder: Can potentially be None (for export-pkg: TODO), in that case
        the conanfile location is used
        """
        # This must be called only after ``layout()`` has been called
        base_folder = conanfile_folder if self.root is None else \
            os.path.normpath(os.path.join(conanfile_folder, self.root))

        self._base_source = base_folder
        self._base_build = output_folder or base_folder
        self._base_generators = output_folder or base_folder
        self._base_export_sources = output_folder or base_folder
        self._base_recipe_metadata = os.path.join(base_folder, "metadata")
        # TODO: It is likely that this base_pkg_metadata is not really used with this value
        self._base_pkg_metadata = output_folder or base_folder

    @property
    def source_folder(self):
        if self._base_source is None:
            return None
        if not self.source:
            return os.path.normpath(self._base_source)

        return os.path.normpath(os.path.join(self._base_source, self.source))

    @property
    def base_source(self):
        return self._base_source

    def set_base_source(self, folder):
        self._base_source = folder

    @property
    def build_folder(self):
        if self._base_build is None:
            return None
        if not self.build:
            return os.path.normpath(self._base_build)
        return os.path.normpath(os.path.join(self._base_build, self.build))

    @property
    def recipe_metadata_folder(self):
        return self._base_recipe_metadata

    def set_base_recipe_metadata(self, folder):
        self._base_recipe_metadata = folder

    @property
    def package_metadata_folder(self):
        return self._base_pkg_metadata

    def set_base_pkg_metadata(self, folder):
        self._base_pkg_metadata = folder

    @property
    def base_build(self):
        return self._base_build

    def set_base_build(self, folder):
        self._base_build = folder

    @property
    def base_package(self):
        return self._base_package

    def set_base_package(self, folder):
        self._base_package = folder

    @property
    def package_folder(self):
        """For the cache, the package folder is only the base"""
        return self._base_package

    def set_finalize_folder(self, folder):
        self._immutable_package_folder = self.package_folder
        self.set_base_package(folder)

    @property
    def immutable_package_folder(self):
        return self._immutable_package_folder or self.package_folder

    @property
    def generators_folder(self):
        if self._base_generators is None:
            return None
        if not self.generators:
            return os.path.normpath(self._base_generators)
        return os.path.normpath(os.path.join(self._base_generators, self.generators))

    def set_base_generators(self, folder):
        self._base_generators = folder

    @property
    def base_export(self):
        return self._base_export

    def set_base_export(self, folder):
        self._base_export = folder

    @property
    def base_export_sources(self):
        return self._base_export_sources

    def set_base_export_sources(self, folder):
        self._base_export_sources = folder
