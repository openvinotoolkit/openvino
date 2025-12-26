import os
import shutil

from urllib.parse import urlparse
from urllib.request import url2pathname

from conan.api.output import ConanOutput
from conan.internal.cache.home_paths import HomePaths
from conan.internal.rest.file_downloader import FileDownloader
from conan.internal.rest.download_cache import DownloadCache
from conan.internal.errors import AuthenticationException, ForbiddenException, NotFoundException
from conan.errors import ConanException
from conan.internal.util.files import mkdir, set_dirty_context_manager, remove_if_dirty, human_size


class SourcesCachingDownloader:
    """ Class for downloading recipe download() urls
    if the config is active, it can use caching/backup-sources
    """
    def __init__(self, conanfile):
        helpers = getattr(conanfile, "_conan_helpers")
        self._global_conf = helpers.global_conf
        self._file_downloader = FileDownloader(helpers.requester, scope=conanfile.display_name,
                                               source_credentials=True)
        self._home_folder = helpers.home_folder
        self._output = conanfile.output
        self._conanfile = conanfile

    def download(self, urls, file_path,
                 retry, retry_wait, verify_ssl, auth, headers, md5, sha1, sha256):
        download_cache_folder = self._global_conf.get("core.sources:download_cache")
        backups_urls = self._global_conf.get("core.sources:download_urls", check_type=list)
        if not (backups_urls or download_cache_folder) or not sha256:
            # regular, non backup/caching download
            if backups_urls or download_cache_folder:
                self._output.warning("Cannot cache download() without sha256 checksum")
            self._download_from_urls(urls, file_path, retry, retry_wait, verify_ssl, auth, headers,
                                     md5, sha1, sha256)
        else:
            self._caching_download(urls, file_path,
                                   retry, retry_wait, verify_ssl, auth, headers, md5, sha1, sha256,
                                   download_cache_folder, backups_urls)

    def _caching_download(self, urls, file_path,
                          retry, retry_wait, verify_ssl, auth, headers, md5, sha1, sha256,
                          download_cache_folder, backups_urls):
        """
        this download will first check in the local cache, if not there, it will go to the list
        of backup_urls defined by user conf (by default ["origin"]), and iterate it until
        something is found.
        """
        # We are going to use the download_urls definition for backups
        download_cache_folder = download_cache_folder or HomePaths(self._home_folder).default_sources_backup_folder
        # regular local shared download cache, not using Conan backup sources servers
        backups_urls = backups_urls or ["origin"]
        if download_cache_folder and not os.path.isabs(download_cache_folder):
            raise ConanException("core.download:download_cache must be an absolute path")

        download_cache = DownloadCache(download_cache_folder)
        cached_path = download_cache.source_path(sha256)
        with download_cache.lock(sha256):
            remove_if_dirty(cached_path)

            if os.path.exists(cached_path):
                self._output.info(f"Source {urls} retrieved from local download cache")
            else:
                with set_dirty_context_manager(cached_path):
                    if None in backups_urls:
                        raise ConanException("Trying to download sources from None backup remote."
                                             f" Remotes were: {backups_urls}")
                    for backup_url in backups_urls:
                        is_last = backup_url is backups_urls[-1]
                        if backup_url == "origin":  # recipe defined URLs
                            if self._origin_download(urls, cached_path, retry, retry_wait,
                                                     verify_ssl, auth, headers, md5, sha1, sha256,
                                                     is_last):
                                break
                        else:
                            if self._backup_download(backup_url, backups_urls, sha256, cached_path,
                                                     urls, is_last):
                                break

            download_cache.update_backup_sources_json(cached_path, self._conanfile, urls)
            # Everything good, file in the cache, just copy it to final destination
            mkdir(os.path.dirname(file_path))
            shutil.copy2(cached_path, file_path)

    def _origin_download(self, urls, cached_path, retry, retry_wait,
                         verify_ssl, auth, headers, md5, sha1, sha256, is_last):
        """ download from the internet, the urls provided by the recipe (mirrors).
        """
        try:
            self._download_from_urls(urls, cached_path, retry, retry_wait,
                                     verify_ssl, auth, headers, md5, sha1, sha256)
        except ConanException as e:
            if is_last:
                raise
            else:
                # TODO: Improve printing of AuthenticationException
                self._output.warning(f"Sources for {urls} failed in 'origin': {e}")
                self._output.warning("Checking backups")
        else:
            if not is_last:
                self._output.info(f"Sources for {urls} found in origin")
            return True

    def _backup_download(self, backup_url, backups_urls, sha256, cached_path, urls, is_last):
        """ download from a Conan backup sources file server, like an Artifactory generic repo
        All failures are bad, except NotFound. The server must be live, working and auth, we
        don't want silently skipping a backup because it is down.
        """
        try:
            backup_url = backup_url if backup_url.endswith("/") else backup_url + "/"
            self._file_downloader.download(backup_url + sha256, cached_path, sha256=sha256)
            self._file_downloader.download(backup_url + sha256 + ".json", cached_path + ".json")
            self._output.info(f"Sources for {urls} found in remote backup {backup_url}")
            return True
        except NotFoundException:
            if is_last:
                raise NotFoundException(f"File {urls} not found in {backups_urls}")
            else:
                self._output.warning(f"File {urls} not found in {backup_url}")
        except (AuthenticationException, ForbiddenException) as e:
            raise ConanException(f"The source backup server '{backup_url}' "
                                 f"needs authentication: {e}. "
                                 f"Please provide 'source_credentials.json'")

    def _download_from_urls(self, urls, file_path, retry, retry_wait, verify_ssl, auth, headers,
                            md5, sha1, sha256):
        """ iterate the recipe provided list of urls (mirrors, all with same checksum) until
        one succeed
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # filename in subfolder must exist
        if not isinstance(urls, (list, tuple)):
            urls = [urls]
        for url in urls:
            try:
                if url.startswith("file:"):  # plain copy from local disk, no real download
                    file_origin = url2pathname(urlparse(url).path)
                    shutil.copyfile(file_origin, file_path)
                    self._file_downloader.check_checksum(file_path, md5, sha1, sha256)
                else:
                    self._file_downloader.download(url, file_path, retry, retry_wait, verify_ssl,
                                                   auth, True, headers, md5, sha1, sha256)
                return  # Success! Return to caller
            except Exception as error:
                if url != urls[-1]:  # If it is not the last one, do not raise, warn and move to next
                    msg = f"Could not download from the URL {url}: {error}."
                    self._output.warning(msg)
                    self._output.info("Trying another mirror.")
                else:
                    raise


class ConanInternalCacheDownloader:
    """ This is used for the download of Conan packages from server, not for sources/backup sources
    """
    def __init__(self, requester, config, scope=None):
        self._download_cache = config.get("core.download:download_cache")
        if self._download_cache and not os.path.isabs(self._download_cache):
            raise ConanException("core.download:download_cache must be an absolute path")
        self._file_downloader = FileDownloader(requester, scope=scope)
        self._scope = scope

    def download(self, url, file_path, auth, verify_ssl, retry, retry_wait, metadata=False):
        if not self._download_cache or metadata:  # Metadata not cached and can be overwritten
            self._file_downloader.download(url, file_path, retry=retry, retry_wait=retry_wait,
                                           verify_ssl=verify_ssl, auth=auth, overwrite=metadata)
            return

        download_cache = DownloadCache(self._download_cache)
        cached_path, h = download_cache.cached_path(url)
        with download_cache.lock(h):
            remove_if_dirty(cached_path)

            if not os.path.exists(cached_path):
                with set_dirty_context_manager(cached_path):
                    self._file_downloader.download(url, cached_path, retry=retry,
                                                   retry_wait=retry_wait, verify_ssl=verify_ssl,
                                                   auth=auth, overwrite=False)
            else:  # Found in cache!
                total_length = os.path.getsize(cached_path)
                is_large_file = total_length > 10000000  # 10 MB
                if is_large_file:
                    base_name = os.path.basename(file_path)
                    hs = human_size(total_length)
                    ConanOutput(scope=self._scope).info(f"Copying {hs} {base_name} from download "
                                                        f"cache, instead of downloading it")

            # Everything good, file in the cache, just copy it to final destination
            mkdir(os.path.dirname(file_path))
            shutil.copy2(cached_path, file_path)
