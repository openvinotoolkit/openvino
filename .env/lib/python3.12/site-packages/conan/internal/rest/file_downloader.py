import os
import re
import time


from conan.api.output import ConanOutput, TimedOutput
from conan.internal.rest import response_to_str
from conan.internal.errors import (ConanConnectionError, RequestErrorException,
                                   AuthenticationException, ForbiddenException, NotFoundException)
from conan.errors import ConanException
from conan.internal.util.files import human_size, check_with_algorithm_sum


class FileDownloader:

    def __init__(self, requester, scope=None, source_credentials=None):
        self._output = ConanOutput(scope=scope)
        self._requester = requester
        self._source_credentials = source_credentials

    def download(self, url, file_path, retry=2, retry_wait=0, verify_ssl=True, auth=None,
                 overwrite=False, headers=None, md5=None, sha1=None, sha256=None):
        """ in order to make the download concurrent, the folder for file_path MUST exist
        """
        assert file_path, "Conan 2.0 always downloads files to disk, not to memory"
        assert os.path.isabs(file_path), "Target file_path must be absolute"

        if os.path.exists(file_path):
            if overwrite:
                self._output.warning("file '%s' already exists, overwriting" % file_path)
            else:
                # Should not happen, better to raise, probably we had to remove
                # the dest folder before
                raise ConanException("Error, the file to download already exists: '%s'" % file_path)

        try:
            for counter in range(retry + 1):
                try:
                    self._download_file(url, auth, headers, file_path, verify_ssl)
                    break
                except (NotFoundException, ForbiddenException, AuthenticationException,
                        RequestErrorException):
                    raise
                except ConanException as exc:
                    if counter == retry:
                        raise
                    else:
                        self._output.warning(exc, warn_tag="network")
                        self._output.info(f"Waiting {retry_wait} seconds to retry...")
                        time.sleep(retry_wait)

            self.check_checksum(file_path, md5, sha1, sha256)
            self._output.debug(f"Downloaded {file_path} from {url}")
        except Exception:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

    @staticmethod
    def check_checksum(file_path, md5, sha1, sha256):
        if md5 is not None:
            check_with_algorithm_sum("md5", file_path, md5)
        if sha1 is not None:
            check_with_algorithm_sum("sha1", file_path, sha1)
        if sha256 is not None:
            check_with_algorithm_sum("sha256", file_path, sha256)

    def _download_file(self, url, auth, headers, file_path, verify_ssl, try_resume=False):
        if try_resume and os.path.exists(file_path):
            range_start = os.path.getsize(file_path)
            headers = headers.copy() if headers else {}
            headers["range"] = "bytes={}-".format(range_start)
        else:
            range_start = 0

        try:
            response = self._requester.get(url, stream=True, verify=verify_ssl, auth=auth,
                                           headers=headers,
                                           source_credentials=self._source_credentials)
        except Exception as exc:
            raise ConanException("Error downloading file %s: '%s'" % (url, exc))

        if not response.ok:
            if response.status_code == 404:
                raise NotFoundException("Not found: %s" % url)
            elif response.status_code == 403:
                if auth is None or (hasattr(auth, "bearer") and auth.bearer is None):
                    # TODO: This is a bit weird, why this conversion? Need to investigate
                    raise AuthenticationException(response_to_str(response))
                raise ForbiddenException(response_to_str(response))
            elif response.status_code == 401:
                raise AuthenticationException(response_to_str(response))
            raise ConanException("Error %d downloading file %s" % (response.status_code, url))

        def get_total_length():
            if range_start:
                content_range = response.headers.get("Content-Range", "")
                match = re.match(r"^bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or range_start != int(match.group(1)):
                    raise ConanException("Error in resumed download from %s\n"
                                         "Incorrect Content-Range header %s" % (url, content_range))
                return int(match.group(3))
            else:
                total_size = response.headers.get('Content-Length') or len(response.content)
                return int(total_size)

        try:
            total_length = get_total_length()
            is_large_file = total_length > 10000000  # 10 MB
            base_name = os.path.basename(file_path)

            def msg_format(msg, downloaded):
                perc = int(total_downloaded_size * 100 / total_length)
                return msg + f" {human_size(downloaded)} {perc}% {base_name}"
            timed_output = TimedOutput(10, out=self._output, msg_format=msg_format)

            if is_large_file:
                hs = human_size(total_length)
                action = "Downloading" if range_start == 0 else "Continuing download of"
                self._output.info(f"{action} {hs} {base_name}")

            chunk_size = 1024 * 100
            total_downloaded_size = range_start
            mode = "ab" if range_start else "wb"
            with open(file_path, mode) as file_handler:
                for chunk in response.iter_content(chunk_size):
                    file_handler.write(chunk)
                    total_downloaded_size += len(chunk)
                    if is_large_file:
                        timed_output.info("Downloaded", total_downloaded_size)

            gzip = (response.headers.get("content-encoding") == "gzip")
            response.close()
            # it seems that if gzip we don't know the size, cannot resume and shouldn't raise
            if total_downloaded_size != total_length and not gzip:
                if (total_length > total_downloaded_size > range_start
                        and response.headers.get("Accept-Ranges") == "bytes"):
                    self._download_file(url, auth, headers, file_path, verify_ssl, try_resume=True)
                else:
                    raise ConanException("Transfer interrupted before complete: %s < %s"
                                         % (total_downloaded_size, total_length))
        except Exception as e:
            # If this part failed, it means problems with the connection to server
            raise ConanConnectionError("Download failed, check server, possibly try again\n%s"
                                       % str(e))
