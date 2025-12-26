import time
from multiprocessing.pool import ThreadPool
from typing import Optional, List

from conan.api.model import Remote, PackagesList
from conan.api.output import ConanOutput
from conan.internal.conan_app import ConanBasicApp
from conan.errors import ConanException
from conan.api.model import PkgReference
from conan.api.model import RecipeReference


class DownloadAPI:
    """ This API is used to download recipes and packages from a remote server."""

    def __init__(self, conan_api):
        self._conan_api = conan_api

    def recipe(self, ref: RecipeReference, remote: Remote, metadata: Optional[List[str]] = None):
        """Download the recipe specified in the ref from the remote.
        If the recipe is already in the cache it will be skipped,
        but the specified metadata will be downloaded."""
        output = ConanOutput()
        app = ConanBasicApp(self._conan_api)
        assert ref.revision, f"Reference '{ref}' must have revision"
        try:
            app.cache.recipe_layout(ref)  # raises if not found
        except ConanException:
            pass
        else:
            output.info(f"Skip recipe {ref.repr_notime()} download, already in cache")
            if metadata:
                app.remote_manager.get_recipe_metadata(ref, remote, metadata)
            return False

        output.info(f"Downloading recipe '{ref.repr_notime()}'")
        if ref.timestamp is None:  # we didnt obtain the timestamp before (in general it should be)
            # Respect the timestamp of the server, the ``get_recipe()`` doesn't do it internally
            # Best would be that ``get_recipe()`` returns the timestamp in the same call
            server_ref = app.remote_manager.get_recipe_revision(ref, remote)
            assert server_ref == ref
            ref.timestamp = server_ref.timestamp
        app.remote_manager.get_recipe(ref, remote, metadata)

        # Download the sources too, don't be lazy
        output.info(f"Downloading '{str(ref)}' sources")
        recipe_layout = app.cache.recipe_layout(ref)
        app.remote_manager.get_recipe_sources(ref, recipe_layout, remote)
        return True

    def package(self, pref: PkgReference, remote: Remote, metadata: Optional[List[str]] = None):
        """Download the package specified in the pref from the remote.
        The recipe for this package binary must already exist in the cache.
        If the package is already in the cache it will be skipped,
        but the specified metadata will be downloaded."""
        output = ConanOutput()
        app = ConanBasicApp(self._conan_api)
        try:
            app.cache.recipe_layout(pref.ref)  # raises if not found
        except ConanException:
            raise ConanException("The recipe of the specified package "
                                 "doesn't exist, download it first")

        skip_download = app.cache.exists_prev(pref)
        if skip_download:
            output.info(f"Skip package {pref.repr_notime()} download, already in cache")
            if metadata:
                app.remote_manager.get_package_metadata(pref, remote, metadata)
            return False

        if pref.timestamp is None:  # we didn't obtain the timestamp before (in general it should be)
            # Respect the timestamp of the server
            server_pref = app.remote_manager.get_package_revision(pref, remote)
            assert server_pref == pref
            pref.timestamp = server_pref.timestamp

        output.info(f"Downloading package '{pref.repr_notime()}'")
        app.remote_manager.get_package(pref, remote, metadata)
        return True

    def download_full(self, package_list: PackagesList, remote: Remote,
                      metadata: Optional[List[str]] = None):
        """Download the recipes and packages specified in the ``package_list`` from the remote,
        parallelized based on ``core.download:parallel``"""
        def _download_pkglist(pkglist):
            for ref, packages in pkglist.items():
                self.recipe(ref, remote, metadata)
                ref_dict = pkglist.recipe_dict(ref)
                ref_dict.pop("files", None)
                ref_dict.pop("upload-urls", None)
                for pref in packages:
                    self.package(pref, remote, metadata)
                    pkg_dict = pkglist.package_dict(pref)
                    pkg_dict.pop("files", None)
                    pkg_dict.pop("upload-urls", None)

        t = time.time()
        parallel = self._conan_api.config.get("core.download:parallel", default=1, check_type=int)
        thread_pool = ThreadPool(parallel) if parallel > 1 else None
        if not thread_pool or len(package_list._data) <= 1:  # FIXME: Iteration when multiple rrevs
            _download_pkglist(package_list)
        else:
            ConanOutput().subtitle(f"Downloading with {parallel} parallel threads")
            thread_pool.map(_download_pkglist, package_list.split())

        if thread_pool:
            thread_pool.close()
            thread_pool.join()

        elapsed = time.time() - t
        ConanOutput().success(f"Download completed in {int(elapsed)}s\n")
