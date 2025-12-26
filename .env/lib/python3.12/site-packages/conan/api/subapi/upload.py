import os
import time
from multiprocessing.pool import ThreadPool
from typing import List

from conan.api.model import PackagesList, Remote
from conan.api.output import ConanOutput
from conan.internal.api.upload import add_urls
from conan.internal.conan_app import ConanApp
from conan.internal.api.uploader import PackagePreparator, UploadExecutor, UploadUpstreamChecker, \
    gather_metadata
from conan.internal.rest.pkg_sign import PkgSignaturesPlugin
from conan.internal.rest.file_uploader import FileUploader
from conan.internal.errors import AuthenticationException, ForbiddenException
from conan.errors import ConanException


class UploadAPI:
    """ This API is used to upload recipes and packages to a remote server."""

    def __init__(self, conan_api, api_helpers):
        self._conan_api = conan_api
        self._api_helpers = api_helpers

    def check_upstream(self, package_list: PackagesList, remote: Remote,
                       enabled_remotes: List[Remote], force=False):
        """ Checks ``remote`` for the existence of the recipes and packages in ``package_list``.
        Items that are not present in the remote will add an ``upload`` key to the entry
        with the value ``True``.

        If the recipe has an upload policy of ``skip``, it will be discarded from the upload list.

        :parameter package_list: A ``PackagesList`` object with the recipes and packages to check.
        :parameter remote: Remote to check.
        :parameter enabled_remotes: List of enabled remotes. This is used to possibly load
            python_requires from the listed recipes if necessary.
        :parameter force: If ``True``, it will skip the check and mark that all items need to be
            uploaded. A ``force_upload`` key will be added to the entries that will be uploaded.
        """
        app = ConanApp(self._conan_api)
        for ref, _ in package_list.items():
            layout = app.cache.recipe_layout(ref)
            conanfile_path = layout.conanfile()
            conanfile = app.loader.load_basic(conanfile_path, remotes=enabled_remotes)
            if conanfile.upload_policy == "skip":
                ConanOutput().info(f"{ref}: Skipping upload of binaries, "
                                   "because upload_policy='skip'")
                package_list.recipe_dict(ref)["packages"] = {}

        UploadUpstreamChecker(app).check(package_list, remote, force)

    def prepare(self, package_list: PackagesList, enabled_remotes: List[Remote],
                metadata: List[str] = None):
        """Compress the recipes and packages and fill the upload_data objects
        with the complete information. It doesn't perform the upload nor checks upstream to see
        if the recipe is still there

        :param package_list: A PackagesList object with the recipes and packages to upload.
        :param enabled_remotes: A list of remotes that are enabled in the client.
            Recipe sources will attempt to be fetched from these remotes.
        :param metadata: A list of patterns of metadata that should be uploaded.
            Default ``None`` means all metadata will be uploaded together with the package artifacts.
            If metadata contains an empty string (``""``),
            it means that no metadata files should be uploaded."""
        if metadata and metadata != [''] and '' in metadata:
            raise ConanException("Empty string and patterns can not be mixed for metadata.")
        app = ConanApp(self._conan_api)
        preparator = PackagePreparator(app, self._api_helpers.global_conf)
        preparator.prepare(package_list, enabled_remotes)
        if metadata != ['']:
            gather_metadata(package_list, app.cache, metadata)
        signer = PkgSignaturesPlugin(app.cache, app.cache_folder)
        # This might add files entries to package_list with signatures
        signer.sign(package_list)

    def _upload(self, package_list, remote):
        app = ConanApp(self._conan_api)
        app.remote_manager.check_credentials(remote)
        executor = UploadExecutor(app)
        executor.upload(package_list, remote)

    def upload_full(self, package_list: PackagesList, remote: Remote, enabled_remotes: List[Remote],
                    check_integrity=False, force=False, metadata: List[str] = None, dry_run=False):
        """ Does the whole process of uploading, including the possibility of parallelizing
        per recipe based on the ``core.upload:parallel`` conf.

        The steps that this method performs are:
            - calls ``conan_api.cache.check_integrity`` to ensure the packages are not corrupted
            - checks the upload policy of the recipes
                - (if it is ``"skip"``, it will not upload the binaries, but will still upload
                  the metadata)
            - checks which revisions already exist in the server so that it can skip the upload
            - prepares the artifacts to upload (compresses the conan_package.tgz)
            - executes the actual upload
            - uploads associated sources backups if any

        :param package_list: A PackagesList object with the recipes and packages to upload.
        :param remote: The remote to upload the packages to.
        :param enabled_remotes: A list of remotes that are enabled in the client.
            Recipe sources will attempt to be fetched from these remotes,
            and to possibly load python_requires from the listed recipes if necessary.
        :param check_integrity: If ``True``, it will check the integrity of the cache packages
            before uploading them. This is useful to ensure that the packages are not corrupted.
        :param force: If ``True``, it will force the upload of the recipes and packages,
            even if they already exist in the remote. Note that this might update the timestamps
        :param metadata: A list of patterns of metadata that should be uploaded.
            Default ``None`` means all metadata will be uploaded together with the package artifacts.
            If metadata contains an empty string (``""``),
            it means that no metadata files should be uploaded.
        :param dry_run: If ``True``, it will not perform the actual upload,
            but will still prepare the artifacts and check the upstream.
        """

        def _upload_pkglist(pkglist, subtitle=lambda _: None):
            if check_integrity:
                subtitle("Checking integrity of cache packages")
                self._conan_api.cache.check_integrity(pkglist)
            # Check if the recipes/packages are in the remote
            subtitle("Checking server for existing packages")
            self.check_upstream(pkglist, remote, enabled_remotes, force)
            subtitle("Preparing artifacts for upload")
            self.prepare(pkglist, enabled_remotes, metadata)

            if not dry_run:
                subtitle("Uploading artifacts")
                self._upload(pkglist, remote)
                backup_files = self._conan_api.cache.get_backup_sources(pkglist)
                self.upload_backup_sources(backup_files)

        t = time.time()
        ConanOutput().title(f"Uploading to remote {remote.name}")
        parallel = self._conan_api.config.get("core.upload:parallel", default=1, check_type=int)
        thread_pool = ThreadPool(parallel) if parallel > 1 else None
        if not thread_pool or len(package_list._data) <= 1:  # FIXME: Iteration when multiple rrevs
            _upload_pkglist(package_list, subtitle=ConanOutput().subtitle)
        else:
            ConanOutput().subtitle(f"Uploading with {parallel} parallel threads")
            thread_pool.map(_upload_pkglist, package_list.split())
        if thread_pool:
            thread_pool.close()
            thread_pool.join()
        elapsed = time.time() - t
        ConanOutput().success(f"Upload completed in {int(elapsed)}s\n")
        add_urls(package_list, remote)

    def upload_backup_sources(self, files: List) -> None:
        """
        Upload to the server the backup sources files, that have been typically gathered by
        CacheAPI.get_backup_sources()

        :param files: The list of files that must be uploaded
        """
        config = self._api_helpers.global_conf
        url = config.get("core.sources:upload_url", check_type=str)
        if url is None:
            return
        url = url if url.endswith("/") else url + "/"

        output = ConanOutput()
        output.subtitle("Uploading backup sources")
        if not files:
            output.info("No backup sources files to upload")
            return

        requester = self._api_helpers.requester
        uploader = FileUploader(requester, verify=True, config=config, source_credentials=True)
        # TODO: For Artifactory, we can list all files once and check from there instead
        #  of 1 request per file, but this is more general
        for file in files:
            basename = os.path.basename(file)
            full_url = url + basename
            is_summary = file.endswith(".json")
            file_kind = "summary" if is_summary else "file"
            try:
                if is_summary or not uploader.exists(full_url, auth=None):
                    output.info(f"Uploading {file_kind} '{basename}' to backup sources server")
                    uploader.upload(full_url, file, dedup=False, auth=None)
                else:
                    output.info(f"File '{basename}' already in backup sources server, "
                                "skipping upload")
            except (AuthenticationException, ForbiddenException) as e:
                if is_summary:
                    output.warning(f"Could not update summary '{basename}' in backup sources server. "
                                   "Skipping updating file but continuing with upload. "
                                   f"Missing permissions?: {e}")
                else:
                    raise ConanException(f"The source backup server '{url}' needs authentication"
                                         f"/permissions, please provide 'source_credentials.json': {e}")

        output.success("Upload backup sources complete\n")
