import copy
import fnmatch
import hashlib
import json
import os
from threading import Thread

from requests.auth import AuthBase, HTTPBasicAuth
from uuid import getnode as get_mac

from conan.api.output import ConanOutput

from conan.internal.rest.caching_file_downloader import ConanInternalCacheDownloader
from conan.internal.rest import response_to_str
from conan.internal.rest.client_routes import ClientV2Router
from conan.internal.rest.file_uploader import FileUploader
from conan.internal.errors import AuthenticationException, ForbiddenException, NotFoundException, \
    RecipeNotFoundException, PackageNotFoundException, EXCEPTION_CODE_MAPPING
from conan.errors import ConanException
from conan.api.model import PkgReference
from conan.internal.paths import EXPORT_SOURCES_TGZ_NAME
from conan.api.model import RecipeReference
from conan.internal.util.dates import from_iso8601_to_timestamp


# TODO: We might want to replace this raw Thread for a ThreadPool, to align with other code usages
class ExceptionThread(Thread):
    def run(self):
        self._exc = None
        try:
            super().run()
        except Exception as e:
            self._exc = e

    def join(self, timeout=None):
        super().join(timeout=timeout)

    def raise_errors(self):
        if self._exc:
            raise self._exc


class JWTAuth(AuthBase):
    """Attaches JWT Authentication to the given Request object."""

    def __init__(self, token):
        self.bearer = "Bearer %s" % str(token) if token else None

    def __call__(self, request):
        if self.bearer:
            request.headers['Authorization'] = self.bearer
        return request


def get_exception_from_error(error_code):
    tmp = {v: k for k, v in EXCEPTION_CODE_MAPPING.items()  # All except NotFound
           if k not in (RecipeNotFoundException, PackageNotFoundException)}
    if error_code in tmp:
        # logger.debug("REST ERROR: %s" % str(tmp[error_code]))
        return tmp[error_code]
    else:
        base_error = int(str(error_code)[0] + "00")
        # logger.debug("REST ERROR: %s" % str(base_error))
        try:
            return tmp[base_error]
        except KeyError:
            return None


def _get_mac_digest():  # To avoid re-hashing all the time the same mac
    cached = getattr(_get_mac_digest, "_cached_value", None)
    if cached is not None:
        return cached
    sha1 = hashlib.sha1()
    sha1.update(str(get_mac()).encode())
    cached = str(sha1.hexdigest())
    _get_mac_digest._cached_value = cached
    return cached


class RestV2Methods:

    def __init__(self, remote_url, token, requester, config, verify_ssl, checksum_deploy=False):
        self.remote_url = remote_url
        self.custom_headers = {'X-Client-Anonymous-Id': _get_mac_digest()}
        self.requester = requester
        self._config = config
        self.verify_ssl = verify_ssl
        self._checksum_deploy = checksum_deploy
        self.router = ClientV2Router(self.remote_url.rstrip("/"))
        self.auth = JWTAuth(token)

    @staticmethod
    def _check_error_response(ret):
        if ret.status_code == 401:
            raise AuthenticationException("Wrong user or password")
        # Cannot check content-type=text/html, conan server is doing it wrong
        if not ret.ok or "html>" in str(ret.content):
            raise ConanException("%s\n\nInvalid server response, check remote URL and "
                                 "try again" % str(ret.content))

    def authenticate(self, user, password):
        """Sends user + password to get:
          - A plain response with a regular token (not supported refresh in the remote) and None
        """
        auth = HTTPBasicAuth(user, password)
        url = self.router.common_authenticate()
        # logger.debug("REST: Authenticate to get access_token: %s" % url)
        ret = self.requester.get(url, auth=auth, headers=self.custom_headers,
                                 verify=self.verify_ssl)

        self._check_error_response(ret)
        return ret.content.decode()

    def check_credentials(self, force_auth=False):
        """If token is not valid will raise AuthenticationException.
        User will be asked for new user/pass"""
        url = self.router.common_check_credentials()
        auth = self.auth
        if force_auth and auth.bearer is None:
            auth = JWTAuth("unset")

        # logger.debug("REST: Check credentials: %s" % url)
        ret = self.requester.get(url, auth=auth, headers=self.custom_headers,
                                 verify=self.verify_ssl)
        if ret.status_code != 200:
            ret.charset = "utf-8"  # To be able to access ret.text (ret.content are bytes)
            text = ret.text if ret.status_code != 404 else "404 Not found"
            raise get_exception_from_error(ret.status_code)(text)
        return ret.content.decode()

    def server_capabilities(self, user=None, password=None):
        """Get information about the server: status, version, type and capabilities"""
        url = self.router.ping()
        # logger.debug("REST: ping: %s" % url)
        if user and password:
            # This can happen in "conan remote login" cmd. Instead of empty token, use HttpBasic
            auth = HTTPBasicAuth(user, password)
        else:
            auth = self.auth
        ret = self.requester.get(url, auth=auth, headers=self.custom_headers, verify=self.verify_ssl)

        server_capabilities = ret.headers.get('X-Conan-Server-Capabilities')
        if not server_capabilities and not ret.ok:
            # Old Artifactory might return 401/403 without capabilities, we don't want
            # to cache them #5687, so raise the exception and force authentication
            raise get_exception_from_error(ret.status_code)(response_to_str(ret))
        if server_capabilities is None:
            # Some servers returning 200-ok, even if not valid repo
            raise ConanException(f"Remote {self.remote_url} doesn't seem like a valid Conan remote")

        return [cap.strip() for cap in server_capabilities.split(",") if cap]

    def _get_json(self, url, data=None, headers=None):
        req_headers = self.custom_headers.copy()
        req_headers.update(headers or {})
        if data:  # POST request
            req_headers.update({'Content-type': 'application/json',
                                'Accept': 'application/json'})
            # logger.debug("REST: post: %s" % url)
            response = self.requester.post(url, auth=self.auth, headers=req_headers,
                                           verify=self.verify_ssl,
                                           stream=True,
                                           data=json.dumps(data))
        else:
            # logger.debug("REST: get: %s" % url)
            response = self.requester.get(url, auth=self.auth, headers=req_headers,
                                          verify=self.verify_ssl,
                                          stream=True)

        if response.status_code != 200:  # Error message is text
            response.charset = "utf-8"  # To be able to access ret.text (ret.content are bytes)
            raise get_exception_from_error(response.status_code)(response_to_str(response))

        content = response.content.decode()
        content_type = response.headers.get("Content-Type")
        if content_type != 'application/json' and content_type != 'application/json; charset=utf-8':
            raise ConanException("%s\n\nResponse from remote is not json, but '%s'"
                                 % (content, content_type))

        try:  # This can fail, if some proxy returns 200 and an html message
            result = json.loads(content)
        except Exception:
            raise ConanException("Remote responded with broken json: %s" % content)
        if not isinstance(result, dict):
            raise ConanException("Unexpected server response %s" % result)
        return result

    def upload_recipe(self, ref, files_to_upload):
        if files_to_upload:
            urls = {fn: self.router.recipe_file(ref, fn)
                    for fn in files_to_upload}
            self._upload_files(files_to_upload, urls, str(ref))

    def upload_package(self, pref, files_to_upload):
        urls = {fn: self.router.package_file(pref, fn)
                for fn in files_to_upload}
        self._upload_files(files_to_upload, urls, str(pref))

    def search(self, pattern=None, ignorecase=True):
        """
        the_files: dict with relative_path: content
        """
        url = self.router.search(pattern, ignorecase)
        response = self._get_json(url)["results"]
        # We need to filter the "_/_" user and channel from Artifactory
        ret = []
        for reference in response:
            try:
                ref = RecipeReference.loads(reference)
            except TypeError:
                raise ConanException("Unexpected response from server.\n"
                                     "URL: `{}`\n"
                                     "Expected an iterable, but got {}.".format(url, type(response)))
            if ref.user == "_":
                ref.user = None
            if ref.channel == "_":
                ref.channel = None
            ret.append(ref)
        return ret

    def search_packages(self, ref, list_only):
        """Client is filtering by the query"""
        url = self.router.search_packages(ref, list_only)
        package_infos = self._get_json(url)
        return package_infos

    def _get_file_list_json(self, url):
        data = self._get_json(url)
        # Discarding (.keys()) still empty metadata for files
        # and make sure the paths like metadata/sign/signature are normalized to /
        data["files"] = list(d.replace("\\", "/") for d in data["files"].keys())
        return data

    def get_recipe(self, ref, dest_folder, metadata, only_metadata):
        url = self.router.recipe_snapshot(ref)
        data = self._get_file_list_json(url)
        server_files = data["files"]
        result = {}

        if not only_metadata:
            accepted_files = ["conanfile.py", "conan_export.tgz", "conanmanifest.txt",
                              "metadata/sign"]
            files = [f for f in server_files if any(f.startswith(m) for m in accepted_files)]
            # If we didn't indicated reference, server got the latest, use absolute now, it's safer
            urls = {fn: self.router.recipe_file(ref, fn) for fn in files}
            self._download_and_save_files(urls, dest_folder, files, parallel=True)
            result.update({fn: os.path.join(dest_folder, fn) for fn in files})
        if metadata:
            metadata = [f"metadata/{m}" for m in metadata]
            files = [f for f in server_files if any(fnmatch.fnmatch(f, m) for m in metadata)]
            urls = {fn: self.router.recipe_file(ref, fn) for fn in files}
            self._download_and_save_files(urls, dest_folder, files, parallel=True, metadata=True)
            result.update({fn: os.path.join(dest_folder, fn) for fn in files})
        return result

    def get_recipe_sources(self, ref, dest_folder):
        # If revision not specified, check latest
        assert ref.revision, f"get_recipe_sources() called without revision {ref}"
        url = self.router.recipe_snapshot(ref)
        data = self._get_file_list_json(url)
        files = data["files"]
        if EXPORT_SOURCES_TGZ_NAME not in files:
            return None
        files = [EXPORT_SOURCES_TGZ_NAME, ]

        # If we didn't indicated reference, server got the latest, use absolute now, it's safer
        urls = {fn: self.router.recipe_file(ref, fn) for fn in files}
        self._download_and_save_files(urls, dest_folder, files, scope=str(ref))
        ret = {fn: os.path.join(dest_folder, fn) for fn in files}
        return ret

    def get_package(self, pref, dest_folder, metadata, only_metadata):
        url = self.router.package_snapshot(pref)
        data = self._get_file_list_json(url)
        server_files = data["files"]
        result = {}
        # Download only known files, but not metadata (except sign)
        if not only_metadata:  # Retrieve package first, then metadata
            accepted_files = ["conaninfo.txt", "conan_package.tgz", "conanmanifest.txt",
                              "metadata/sign"]
            files = [f for f in server_files if any(f.startswith(m) for m in accepted_files)]
            # If we didn't indicated reference, server got the latest, use absolute now, it's safer
            urls = {fn: self.router.package_file(pref, fn) for fn in files}
            self._download_and_save_files(urls, dest_folder, files, scope=str(pref.ref))
            result.update({fn: os.path.join(dest_folder, fn) for fn in files})

        if metadata:
            metadata = [f"metadata/{m}" for m in metadata]
            files = [f for f in server_files if any(fnmatch.fnmatch(f, m) for m in metadata)]
            urls = {fn: self.router.package_file(pref, fn) for fn in files}
            self._download_and_save_files(urls, dest_folder, files, scope=str(pref.ref),
                                          metadata=True)
            result.update({fn: os.path.join(dest_folder, fn) for fn in files})
        return result

    def _upload_files(self, files, urls, ref):
        failed = []
        uploader = FileUploader(self.requester, self.verify_ssl, self._config)
        # conan_package.tgz and conan_export.tgz are uploaded first to avoid uploading conaninfo.txt
        # or conanamanifest.txt with missing files due to a network failure
        for filename in sorted(files):
            # As the filenames are sorted, the last one is always "conanmanifest.txt"
            resource_url = urls[filename]
            try:
                uploader.upload(resource_url, files[filename], auth=self.auth,
                                dedup=self._checksum_deploy, ref=ref)
            except (AuthenticationException, ForbiddenException):
                raise
            except Exception as exc:
                ConanOutput().error(f"\nError uploading file: {filename}, '{exc}'",
                                    error_type="exception")
                failed.append(filename)

        if failed:
            raise ConanException("Execute upload again to retry upload the failed files: %s"
                                 % ", ".join(failed))

    def _download_and_save_files(self, urls, dest_folder, files, parallel=False, scope=None,
                                 metadata=False):
        # Take advantage of filenames ordering, so that conan_package.tgz and conan_export.tgz
        # can be < conanfile, conaninfo, and sent always the last, so smaller files go first
        retry = self._config.get("core.download:retry", check_type=int, default=2)
        retry_wait = self._config.get("core.download:retry_wait", check_type=int, default=0)
        downloader = ConanInternalCacheDownloader(self.requester, self._config, scope=scope)
        threads = []

        for filename in sorted(files, reverse=True):
            resource_url = urls[filename]
            abs_path = os.path.join(dest_folder, filename)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)  # filename in subfolder must exist
            if parallel:
                kwargs = {"url": resource_url, "file_path": abs_path, "retry": retry,
                          "retry_wait": retry_wait, "verify_ssl": self.verify_ssl,
                          "auth": self.auth, "metadata": metadata}
                thread = ExceptionThread(target=downloader.download, kwargs=kwargs)
                threads.append(thread)
                thread.start()
            else:
                downloader.download(url=resource_url, file_path=abs_path, auth=self.auth,
                                    verify_ssl=self.verify_ssl, retry=retry, retry_wait=retry_wait,
                                    metadata=metadata)
        for t in threads:
            t.join()
        for t in threads:  # Need to join all before raising errors
            t.raise_errors()

    def remove_all_packages(self, ref):
        """ Remove all packages from the specified reference"""
        self.check_credentials()
        assert ref.revision is not None, "remove_packages needs RREV"

        url = self.router.remove_all_packages(ref)
        response = self.requester.delete(url, auth=self.auth, verify=self.verify_ssl,
                                         headers=self.custom_headers)
        if response.status_code == 404:
            # Double check if it is a 404 because there are no packages
            try:
                package_search_url = self.router.search_packages(ref, list_only=True)
                if not self._get_json(package_search_url):
                    return
            except Exception as e:
                pass
        if response.status_code != 200:  # Error message is text
            # To be able to access ret.text (ret.content are bytes)
            response.charset = "utf-8"
            raise get_exception_from_error(response.status_code)(response.text)

    def remove_packages(self, prefs):
        self.check_credentials()
        for pref in prefs:
            if not pref.revision:
                prevs = self.get_package_revisions_references(pref)
            else:
                prevs = [pref]
            for prev in prevs:
                url = self.router.remove_package(prev)
                response = self.requester.delete(url, auth=self.auth, headers=self.custom_headers,
                                                 verify=self.verify_ssl)
                if response.status_code == 404:
                    raise PackageNotFoundException(pref)
                if response.status_code != 200:  # Error message is text
                    # To be able to access ret.text (ret.content are bytes)
                    response.charset = "utf-8"
                    raise get_exception_from_error(response.status_code)(response.text)

    def remove_recipe(self, ref):
        """ Remove a recipe and packages """
        self.check_credentials()
        if ref.revision is None:
            # Remove all the RREVs
            refs = self.get_recipe_revisions_references(ref)
        else:
            refs = [ref]

        for ref in refs:
            url = self.router.remove_recipe(ref)
            response = self.requester.delete(url, auth=self.auth, headers=self.custom_headers,
                                             verify=self.verify_ssl)
            if response.status_code == 404:
                raise RecipeNotFoundException(ref)
            if response.status_code != 200:  # Error message is text
                # To be able to access ret.text (ret.content are bytes)
                response.charset = "utf-8"
                raise get_exception_from_error(response.status_code)(response.text)

    def get_recipe_revision_reference(self, ref):
        # FIXME: implement this new endpoint in the remotes?
        assert ref.revision, "recipe_exists has to be called with a complete reference"
        ref_without_rev = copy.copy(ref)
        ref_without_rev.revision = None
        try:
            remote_refs = self.get_recipe_revisions_references(ref_without_rev)
        except NotFoundException:
            raise RecipeNotFoundException(ref)
        for r in remote_refs:
            if r == ref:
                return r
        raise RecipeNotFoundException(ref)

    def get_package_revision_reference(self, pref):
        # FIXME: implement this endpoint in the remotes?
        assert pref.revision, "get_package_revision_reference has to be called with a complete reference"
        pref_without_rev = copy.copy(pref)
        pref_without_rev.revision = None
        try:
            remote_prefs = self.get_package_revisions_references(pref_without_rev)
        except NotFoundException:
            raise PackageNotFoundException(pref)
        for p in remote_prefs:
            if p == pref:
                return p
        raise PackageNotFoundException(pref)

    def get_recipe_revisions_references(self, ref):
        assert ref.revision is None
        url = self.router.recipe_revisions(ref)
        tmp = self._get_json(url)["revisions"]
        remote_refs = []
        for item in tmp:
            _tmp = copy.copy(ref)
            _tmp.revision = item.get("revision")
            _tmp.timestamp = from_iso8601_to_timestamp(item.get("time"))
            remote_refs.append(_tmp)
        return remote_refs

    def get_latest_recipe_reference(self, ref):
        url = self.router.recipe_latest(ref)
        data = self._get_json(url)
        remote_ref = copy.copy(ref)
        remote_ref.revision = data.get("revision")
        remote_ref.timestamp = from_iso8601_to_timestamp(data.get("time"))
        return remote_ref

    def get_package_revisions_references(self, pref):
        assert pref.revision is None
        url = self.router.package_revisions(pref)
        tmp = self._get_json(url)["revisions"]
        remote_prefs = [PkgReference(pref.ref, pref.package_id, item.get("revision"),
                        from_iso8601_to_timestamp(item.get("time"))) for item in tmp]
        return remote_prefs

    def get_latest_package_reference(self, pref: PkgReference, headers):
        url = self.router.package_latest(pref)
        data = self._get_json(url, headers=headers)
        remote_pref = copy.copy(pref)
        remote_pref.revision = data.get("revision")
        remote_pref.timestamp = from_iso8601_to_timestamp(data.get("time"))
        return remote_pref
