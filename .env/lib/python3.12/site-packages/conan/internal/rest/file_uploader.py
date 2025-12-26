import io
import os
import time

from conan.api.output import ConanOutput, TimedOutput
from conan.internal.rest import response_to_str
from conan.internal.errors import InternalErrorException, RequestErrorException, AuthenticationException, \
    ForbiddenException, NotFoundException
from conan.errors import ConanException
from conan.internal.util.files import sha1sum


class FileUploader(object):

    def __init__(self, requester, verify, config, source_credentials=None):
        self._requester = requester
        self._config = config
        self._verify_ssl = verify
        self._source_credentials = source_credentials

    @staticmethod
    def _handle_400_response(response, auth):
        if response.status_code == 400:
            raise RequestErrorException(response_to_str(response))

        if response.status_code == 401:
            raise AuthenticationException(response_to_str(response))

        if response.status_code == 403:
            if auth is None or auth.bearer is None:
                raise AuthenticationException(response_to_str(response))
            raise ForbiddenException(response_to_str(response))

    def _dedup(self, url, headers, auth):
        """ send the headers to see if it is possible to skip uploading the file, because it
        is already in the server. Artifactory support file deduplication
        """
        dedup_headers = {"X-Checksum-Deploy": "true"}
        if headers:
            dedup_headers.update(headers)
        response = self._requester.put(url, data="", verify=self._verify_ssl, headers=dedup_headers,
                                       auth=auth, source_credentials=self._source_credentials)
        if response.status_code == 500:
            raise InternalErrorException(response_to_str(response))

        self._handle_400_response(response, auth)

        if response.status_code == 201:  # Artifactory returns 201 if the file is there
            return response

    def exists(self, url, auth):
        response = self._requester.head(url, verify=self._verify_ssl, auth=auth,
                                        source_credentials=self._source_credentials)
        return bool(response.ok)

    def upload(self, url, abs_path, auth=None, dedup=False, retry=None, retry_wait=None,
               ref=None):
        retry = retry if retry is not None else self._config.get("core.upload:retry", default=1,
                                                                 check_type=int)
        retry_wait = retry_wait if retry_wait is not None else \
            self._config.get("core.upload:retry_wait", default=5, check_type=int)

        # Send always the header with the Sha1
        headers = {"X-Checksum-Sha1": sha1sum(abs_path)}
        if dedup:
            response = self._dedup(url, headers, auth)
            if response:
                return response

        for counter in range(retry + 1):
            try:
                return self._upload_file(url, abs_path, headers, auth, ref)
            except (NotFoundException, ForbiddenException, AuthenticationException,
                    RequestErrorException):
                raise
            except ConanException as exc:
                if counter == retry:
                    raise
                else:
                    ConanOutput().warning(exc, warn_tag="network")
                    ConanOutput().info("Waiting %d seconds to retry..." % retry_wait)
                    time.sleep(retry_wait)

    def _upload_file(self, url, abs_path, headers, auth, ref):
        with FileProgress(abs_path, mode='rb', msg=f"{ref}: Uploading") as file_handler:
            try:
                response = self._requester.put(url, data=file_handler, verify=self._verify_ssl,
                                               headers=headers, auth=auth,
                                               source_credentials=self._source_credentials)
                self._handle_400_response(response, auth)
                response.raise_for_status()  # Raise HTTPError for bad http response status
                return response
            except ConanException:
                raise
            except Exception as exc:
                raise ConanException(exc)


class FileProgress(io.FileIO):
    def __init__(self, path: str, msg: str = "Uploading", interval: float = 10, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._size = os.path.getsize(path)
        self._filename = os.path.basename(path)
        # Report only on big sizes (>100MB)
        self._reporter = TimedOutput(interval=interval) if self._size > 100_000_000 else None
        self._bytes_read = 0
        self.msg = msg

    def read(self, size: int = -1) -> bytes:
        block = super().read(size)
        self._bytes_read += len(block)
        if self._reporter:
            current_percentage = int(self._bytes_read * 100.0 / self._size) if self._size != 0 else 0
            self._reporter.info(f"{self.msg} {self._filename}: {current_percentage}%")
        return block
