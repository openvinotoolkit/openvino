import json
import os
import shutil
import tarfile
import tempfile

from conan.api.model import PackagesList
from conan.api.output import ConanOutput
from conan.internal.api.uploader import compress_files
from conan.internal.cache.cache import PkgCache
from conan.internal.cache.conan_reference_layout import (EXPORT_SRC_FOLDER, EXPORT_FOLDER,
                                                         SRC_FOLDER, METADATA,
                                                         DOWNLOAD_EXPORT_FOLDER)
from conan.internal.cache.home_paths import HomePaths
from conan.internal.cache.integrity_check import IntegrityChecker
from conan.internal.rest.download_cache import DownloadCache
from conan.errors import ConanException
from conan.api.model import PkgReference
from conan.api.model import RecipeReference
from conan.internal.util.dates import revision_timestamp_now
from conan.internal.util.files import rmdir, mkdir, remove, save


class CacheAPI:
    """ This CacheAPI is used to interact with the packages storage cache
    """

    def __init__(self, conan_api, api_helpers):
        self._conan_api = conan_api
        self._api_helpers = api_helpers

    def export_path(self, ref: RecipeReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        ref = _resolve_latest_ref(cache, ref)
        ref_layout = cache.recipe_layout(ref)
        return _check_folder_existence(ref, "export", ref_layout.export())

    def recipe_metadata_path(self, ref: RecipeReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        ref = _resolve_latest_ref(cache, ref)
        ref_layout = cache.recipe_layout(ref)
        return _check_folder_existence(ref, "metadata", ref_layout.metadata())

    def export_source_path(self, ref: RecipeReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        ref = _resolve_latest_ref(cache, ref)
        ref_layout = cache.recipe_layout(ref)
        return _check_folder_existence(ref, "export_sources", ref_layout.export_sources())

    def source_path(self, ref: RecipeReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        ref = _resolve_latest_ref(cache, ref)
        ref_layout = cache.recipe_layout(ref)
        return _check_folder_existence(ref, "source", ref_layout.source())

    def build_path(self, pref: PkgReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        pref = _resolve_latest_pref(cache, pref)
        ref_layout = cache.pkg_layout(pref)
        return _check_folder_existence(pref, "build", ref_layout.build())

    def package_metadata_path(self, pref: PkgReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        pref = _resolve_latest_pref(cache, pref)
        ref_layout = cache.pkg_layout(pref)
        return _check_folder_existence(pref, "metadata", ref_layout.metadata())

    def package_path(self, pref: PkgReference):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        pref = _resolve_latest_pref(cache, pref)
        ref_layout = cache.pkg_layout(pref)
        if os.path.exists(ref_layout.finalize()):
            return ref_layout.finalize()
        return _check_folder_existence(pref, "package", ref_layout.package())

    def check_integrity(self, package_list, return_pkg_list=False):
        """
        Check if the recipes and packages are corrupted

        :param package_list: PackagesList to check
        :param return_pkg_list: If True, return a PackagesList with corrupted artifacts
        :return: PackagesList with corrupted artifacts if return_pkg_list is True
        :raises: ConanExcepcion if there are corrupted artifacts and return_pkg_list is False
        """
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        checker = IntegrityChecker(cache)
        corrupted_pkg_list = checker.check(package_list)
        if return_pkg_list:
            return corrupted_pkg_list
        if corrupted_pkg_list:
            raise ConanException("There are corrupted artifacts, check the error logs")

    def clean(self, package_list, source=True, build=True, download=True, temp=True,
              backup_sources=False):
        """
        Remove non critical folders from the cache, like source, build and download (.tgz store)
        folders.

        :param package_list: the package lists that should be cleaned
        :param source: boolean, remove the "source" folder if True
        :param build: boolean, remove the "build" folder if True
        :param download: boolean, remove the "download (.tgz)" folder if True
        :param temp: boolean, remove the temporary folders
        :param backup_sources: boolean, remove the "source" folder if True
        :return:
        """

        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        if temp:
            rmdir(cache.temp_folder)
            # Clean those build folders that didn't succeed to create a package and wont be in DB
            builds_folder = cache.builds_folder
            if os.path.isdir(builds_folder):
                ConanOutput().verbose(f"Cleaning temporary folders")
                for subdir in os.listdir(builds_folder):
                    folder = os.path.join(builds_folder, subdir)
                    manifest = os.path.join(folder, "p", "conanmanifest.txt")
                    info = os.path.join(folder, "p", "conaninfo.txt")
                    if not os.path.exists(manifest) or not os.path.exists(info):
                        rmdir(folder)
        if backup_sources:
            backup_files = self._conan_api.cache.get_backup_sources(package_list, exclude=False,
                                                                    only_upload=False)
            ConanOutput().verbose(f"Cleaning {len(backup_files)} backup sources")
            for f in backup_files:
                remove(f)

        for ref, packages in package_list.items():
            ConanOutput(ref.repr_notime()).verbose("Cleaning recipe cache contents")
            ref_layout = cache.recipe_layout(ref)
            if source:
                rmdir(ref_layout.source())
            if download:
                rmdir(ref_layout.download_export())
            for pref in packages:
                ConanOutput(pref).verbose("Cleaning package cache contents")
                pref_layout = cache.pkg_layout(pref)
                if build:
                    rmdir(pref_layout.build())
                    # It is important to remove the "build_id" identifier if build-folder is removed
                    cache.remove_build_id(pref)
                if download:
                    rmdir(pref_layout.download_package())

    def save(self, package_list: PackagesList, tgz_path, no_source=False) -> None:
        global_conf = self._api_helpers.global_conf
        cache = PkgCache(self._conan_api.cache_folder, global_conf)
        cache_folder = cache.store  # Note, this is not the home, but the actual package cache
        out = ConanOutput()
        mkdir(os.path.dirname(tgz_path))
        compresslevel = global_conf.get("core.gzip:compresslevel", check_type=int)
        tar_files: dict[str, str] = {}  # {path_in_tar: abs_path}

        for ref, packages in package_list.items():
            ref_layout = cache.recipe_layout(ref)
            recipe_folder = os.path.relpath(ref_layout.base_folder, cache_folder)
            recipe_folder = recipe_folder.replace("\\", "/")  # make win paths portable
            ref_bundle = package_list.recipe_dict(ref)
            ref_bundle["recipe_folder"] = recipe_folder
            out.info(f"Saving {ref}: {recipe_folder}")
            # Package only selected folders, not DOWNLOAD one
            for f in (EXPORT_FOLDER, EXPORT_SRC_FOLDER, SRC_FOLDER):
                if f == SRC_FOLDER and no_source:
                    continue
                path = os.path.join(cache_folder, recipe_folder, f)
                if os.path.exists(path):
                    tar_files[f"{recipe_folder}/{f}"] = path
            path = os.path.join(cache_folder, recipe_folder, DOWNLOAD_EXPORT_FOLDER, METADATA)
            if os.path.exists(path):
                tar_files[f"{recipe_folder}/{DOWNLOAD_EXPORT_FOLDER}/{METADATA}"] = path

            for pref in packages:
                pref_layout = cache.pkg_layout(pref)
                pkg_folder = pref_layout.package()
                folder = os.path.relpath(pkg_folder, cache_folder)
                folder = folder.replace("\\", "/")  # make win paths portable
                pkg_dict = package_list.package_dict(pref)
                pkg_dict["package_folder"] = folder
                out.info(f"Saving {pref}: {folder}")
                tar_files[folder] = os.path.join(cache_folder, folder)

                if os.path.exists(pref_layout.metadata()):
                    metadata_folder = os.path.relpath(pref_layout.metadata(), cache_folder)
                    metadata_folder = metadata_folder.replace("\\", "/")  # make paths portable
                    pkg_dict["metadata_folder"] = metadata_folder
                    out.info(f"Saving {pref} metadata: {metadata_folder}")
                    tar_files[metadata_folder] = os.path.join(cache_folder, metadata_folder)

        # Create a temporary file in order to reuse compress_files functionality
        serialized = json.dumps(package_list.serialize(), indent=2)
        pkglist_path = os.path.join(tempfile.gettempdir(), "pkglist.json")
        save(pkglist_path, serialized)
        tar_files["pkglist.json"] = pkglist_path
        compress_files(tar_files, os.path.basename(tgz_path), os.path.dirname(tgz_path),
                       compresslevel, recursive=True)
        remove(pkglist_path)

    def restore(self, path) -> PackagesList:
        if not os.path.isfile(path):
            raise ConanException(f"Restore archive doesn't exist in {path}")

        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        cache_folder = cache.store  # Note, this is not the home, but the actual package cache

        with open(path, mode='rb') as file_handler:
            the_tar = tarfile.open(fileobj=file_handler)
            fileobj = the_tar.extractfile("pkglist.json")
            pkglist = fileobj.read()
            the_tar.extraction_filter = (lambda member, _: member)  # fully_trusted (Py 3.14)
            the_tar.extractall(path=cache_folder)
            the_tar.close()

        # After unzipping the files, we need to update the DB that references these files
        out = ConanOutput()
        package_list = PackagesList.deserialize(json.loads(pkglist))
        for ref, packages in package_list.items():
            ref_bundle = package_list.recipe_dict(ref)
            ref.timestamp = revision_timestamp_now()
            ref_bundle["timestamp"] = ref.timestamp
            try:
                recipe_layout = cache.recipe_layout(ref)
            except ConanException:
                recipe_layout = cache.create_ref_layout(ref)  # new DB folder entry
            recipe_folder = ref_bundle["recipe_folder"]
            rel_path = os.path.relpath(recipe_layout.base_folder, cache_folder)
            rel_path = rel_path.replace("\\", "/")
            # In the case of recipes, they are always "in place", so just checking it
            assert rel_path == recipe_folder, f"{rel_path}!={recipe_folder}"
            out.info(f"Restore: {ref} in {recipe_folder}")
            for pref in packages:
                pref.timestamp = revision_timestamp_now()
                pref_bundle = package_list.package_dict(pref)
                pref_bundle["timestamp"] = pref.timestamp
                try:
                    pkg_layout = cache.pkg_layout(pref)
                except ConanException:
                    pkg_layout = cache.create_pkg_layout(pref)  # DB Folder entry
                # FIXME: This is not taking into account the existence of previous package
                unzipped_pkg_folder = pref_bundle["package_folder"]
                out.info(f"Restore: {pref} in {unzipped_pkg_folder}")
                # If the DB folder entry is different to the disk unzipped one, we need to move it
                # This happens for built (not downloaded) packages in the source "conan cache save"
                db_pkg_folder = os.path.relpath(pkg_layout.package(), cache_folder)
                db_pkg_folder = db_pkg_folder.replace("\\", "/")
                if db_pkg_folder != unzipped_pkg_folder:
                    # If a previous package exists, like a previous restore, then remove it
                    if os.path.exists(pkg_layout.package()):
                        shutil.rmtree(pkg_layout.package())
                    shutil.move(os.path.join(cache_folder, unzipped_pkg_folder),
                                pkg_layout.package())
                    pref_bundle["package_folder"] = db_pkg_folder
                unzipped_metadata_folder = pref_bundle.get("metadata_folder")
                if unzipped_metadata_folder:
                    # FIXME: Restore metadata is not incremental, but destructive
                    out.info(f"Restore: {pref} metadata in {unzipped_metadata_folder}")
                    db_metadata_folder = os.path.relpath(pkg_layout.metadata(), cache_folder)
                    db_metadata_folder = db_metadata_folder.replace("\\", "/")
                    if db_metadata_folder != unzipped_metadata_folder:
                        # We need to put the package in the final location in the cache
                        if os.path.exists(pkg_layout.metadata()):
                            shutil.rmtree(pkg_layout.metadata())
                        shutil.move(os.path.join(cache_folder, unzipped_metadata_folder),
                                    pkg_layout.metadata())
                        pref_bundle["metadata_folder"] = db_metadata_folder

        return package_list

    def get_backup_sources(self, package_list=None, exclude=True, only_upload=True):
        """Get list of backup source files currently present in the cache,
        either all of them if no argument, or filtered by those belonging to the references
        in the package_list

        :param package_list: a PackagesList object to filter backup files from (The files should
          have been downloaded form any of the references in the package_list)
        :param exclude: if True, exclude the sources that come from URLs present the
          core.sources:exclude_urls global conf
        :param only_upload: if True, only return the files for packages that are set to be uploaded
        """
        config = self._api_helpers.global_conf
        download_cache_path = config.get("core.sources:download_cache")
        download_cache_path = download_cache_path or HomePaths(
            self._conan_api.cache_folder).default_sources_backup_folder
        excluded_urls = config.get("core.sources:exclude_urls",
                                   check_type=list, default=[]) if exclude else []
        download_cache = DownloadCache(download_cache_path)
        return download_cache.get_backup_sources_files(excluded_urls, package_list, only_upload)

    def path_to_ref(self, path):
        cache = PkgCache(self._conan_api.cache_folder, self._api_helpers.global_conf)
        result = cache.path_to_ref(path)
        if result is None:
            base, folder = os.path.split(path)
            result = cache.path_to_ref(base)
        return result


def _resolve_latest_ref(cache, ref):
    if ref.revision is None or ref.revision == "latest":
        ref.revision = None
        result = cache.get_latest_recipe_revision(ref)
        if result is None:
            raise ConanException(f"'{ref}' not found in cache")
        ref = result
    return ref


def _resolve_latest_pref(cache, pref):
    pref.ref = _resolve_latest_ref(cache, pref.ref)
    if pref.revision is None or pref.revision == "latest":
        pref.revision = None
        result = cache.get_latest_package_revision(pref)
        if result is None:
            raise ConanException(f"'{pref.repr_notime()}' not found in cache")
        pref = result
    return pref


def _check_folder_existence(ref, folder_name, folder_path):
    if not os.path.exists(folder_path):
        raise ConanException(f"'{folder_name}' folder does not exist for the reference {ref}")
    return folder_path
