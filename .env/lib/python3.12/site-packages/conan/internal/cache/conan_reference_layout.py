import os
from contextlib import contextmanager

from conan.internal.model.manifest import FileTreeManifest
from conan.internal.paths import CONANFILE, DATA_YML
from conan.internal.util.files import set_dirty, clean_dirty, is_dirty, rmdir


# To be able to change them later to something shorter
SRC_FOLDER = "s"
BUILD_FOLDER = "b"
PACKAGES_FOLDER = "p"
FINALIZE_FOLDER = "f"
EXPORT_FOLDER = "e"
EXPORT_SRC_FOLDER = "es"
DOWNLOAD_EXPORT_FOLDER = "d"
METADATA = "metadata"


class LayoutBase:
    def __init__(self, ref, base_folder):
        self.reference = ref
        self._base_folder = base_folder

    @property
    def base_folder(self):
        return self._base_folder

    def remove(self):
        rmdir(self._base_folder)


class BasicLayout(LayoutBase):
    # For editables and platform_requires
    def conanfile(self):
        # the full conanfile path (including other other.py names) for editables
        # None for platform_requires
        return self._base_folder

    def metadata(self):
        if self._base_folder is None:
            return
        return os.path.join(os.path.dirname(self._base_folder), METADATA)


class RecipeLayout(LayoutBase):

    @contextmanager
    def conanfile_write_lock(self, output):
        yield

    def export(self):
        return os.path.join(self._base_folder, EXPORT_FOLDER)

    def export_sources(self):
        return os.path.join(self._base_folder, EXPORT_SRC_FOLDER)

    def metadata(self):
        return os.path.join(self.download_export(), METADATA)

    def download_export(self):
        return os.path.join(self._base_folder, DOWNLOAD_EXPORT_FOLDER)

    def source(self):
        return os.path.join(self._base_folder, SRC_FOLDER)

    def conanfile(self):
        return os.path.join(self.export(), CONANFILE)

    def conandata(self):
        return os.path.join(self.export(), DATA_YML)

    def recipe_manifests(self):
        # Used for comparison and integrity check
        export_folder = self.export()
        readed_manifest = FileTreeManifest.load(export_folder)
        exports_source_folder = self.export_sources()
        expected_manifest = FileTreeManifest.create(export_folder, exports_source_folder)
        return readed_manifest, expected_manifest

    def sources_remove(self):
        src_folder = self.source()
        rmdir(src_folder)


class PackageLayout(LayoutBase):

    def __init__(self, ref, base_folder):
        super().__init__(ref, base_folder)
        self.build_id = None

    # TODO: cache2.0 locks implementation
    @contextmanager
    def package_lock(self):
        yield

    def build(self):
        return os.path.join(self._base_folder, BUILD_FOLDER)

    def package(self):
        return os.path.join(self._base_folder, PACKAGES_FOLDER)

    def finalize(self):
        return os.path.join(self._base_folder, FINALIZE_FOLDER)

    def download_package(self):
        return os.path.join(self._base_folder, DOWNLOAD_EXPORT_FOLDER)

    def metadata(self):
        return os.path.join(self.download_package(), METADATA)

    def package_manifests(self):
        package_folder = self.package()
        readed_manifest = FileTreeManifest.load(package_folder)
        expected_manifest = FileTreeManifest.create(package_folder)
        return readed_manifest, expected_manifest

    @contextmanager
    def set_dirty_context_manager(self):
        set_dirty(self.package())
        yield
        clean_dirty(self.package())

    # TODO: cache2.0 check this
    def package_is_dirty(self):
        return is_dirty(self.package())

    def build_remove(self):
        rmdir(self.build())

    # TODO: cache2.0 locks
    def package_remove(self):
        # Here we could validate and check we own a write lock over this package
        tgz_folder = self.download_package()
        rmdir(tgz_folder)
        rmdir(self.package())
        if is_dirty(self.package()):
            clean_dirty(self.package())
