import gzip
import os
import stat
import platform
import shutil
import subprocess
from typing import Optional
from contextlib import contextmanager
from fnmatch import fnmatch
from shutil import which


from conan.internal.rest.caching_file_downloader import SourcesCachingDownloader
from conan.errors import ConanException
from conan.internal.rest.file_uploader import FileProgress
from conan.internal.util.files import rmdir as _internal_rmdir, human_size, check_with_algorithm_sum


def load(conanfile, path, encoding="utf-8"):
    """
    Utility function to load files in one line. It will manage the open and close of the file,
    and load binary encodings. Returns the content of the file.

    :param conanfile: The current recipe object. Always use ``self``.
    :param path: Path to the file to read
    :param encoding: (Optional, Defaulted to ``utf-8``): Specifies the input file text encoding.
    :return: The contents of the file
    """
    with open(path, "r", encoding=encoding, newline="") as handle:
        tmp = handle.read()
        return tmp


def save(conanfile, path, content, append=False, encoding="utf-8"):
    """
    Utility function to save files in one line. It will manage the open and close of the file
    and creating directories if necessary.

    :param conanfile: The current recipe object. Always use ``self``.
    :param path: Path of the file to be created.
    :param content: Content (str or bytes) to be write to the file.
    :param append: (Optional, Defaulted to False): If ``True`` the contents will be appended to the
           existing one.
    :param encoding: (Optional, Defaulted to utf-8): Specifies the output file text encoding.
    """
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "a" if append else "w", encoding=encoding, newline="") as handle:
        handle.write(content)


def mkdir(conanfile, path):
    """
    Utility functions to create a directory. The existence of the specified directory is checked,
    so mkdir() will do nothing if the directory already exists.

    :param conanfile: The current recipe object. Always use ``self``.
    :param path: Path to the folder to be created.
    """
    if os.path.exists(path):
        return
    os.makedirs(path)


def rmdir(conanfile, path):
    _internal_rmdir(path)


def rm(conanfile, pattern, folder, recursive=False, excludes=None):
    """
    Utility functions to remove files matching a ``pattern`` in a ``folder``.

    :param conanfile: The current recipe object. Always use ``self``.
    :param pattern: Pattern that the files to be removed have to match (fnmatch).
    :param folder: Folder to search/remove the files.
    :param recursive: If ``recursive`` is specified it will search in the subfolders.
    :param excludes: (Optional, defaulted to None) A tuple/list of fnmatch patterns or even a
                     single one to be excluded from the remove pattern.
    """
    if excludes and not isinstance(excludes, (tuple, list)):
        excludes = (excludes,)
    elif not excludes:
        excludes = []

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if fnmatch(filename, pattern) and not any(fnmatch(filename, it) for it in excludes):
                fullname = os.path.join(root, filename)
                os.unlink(fullname)
        if not recursive:
            break


def get(conanfile, url, md5=None, sha1=None, sha256=None, destination=".", filename="",
        keep_permissions=False, pattern=None, verify=True, retry=None, retry_wait=None,
        auth=None, headers=None, strip_root=False, extract_filter=None, excludes=None):
    """
    High level download and decompressing of a tgz, zip or other compressed format file.
    Just a high level wrapper for download, unzip, and remove the temporary zip file once unzipped.
    You can pass hash checking parameters: ``md5``, ``sha1``, ``sha256``. All the specified
    algorithms will be checked. If any of them doesn't match, it will raise a ``ConanException``.

    :param conanfile: The current recipe object. Always use ``self``.
    :param destination: (Optional defaulted to ``.``) Destination folder
    :param filename: (Optional defaulted to '') If provided, the saved file will have the specified name,
           otherwise it is deduced from the URL
    :param url: forwarded to ``tools.file.download()``.
    :param md5: forwarded to ``tools.file.download()``.
    :param sha1:  forwarded to ``tools.file.download()``.
    :param sha256:  forwarded to ``tools.file.download()``.
    :param keep_permissions:  forwarded to ``tools.file.unzip()``.
    :param pattern: forwarded to ``tools.file.unzip()``.
    :param verify:  forwarded to ``tools.file.download()``.
    :param retry:  forwarded to ``tools.file.download()``.
    :param retry_wait: S forwarded to ``tools.file.download()``.
    :param auth:  forwarded to ``tools.file.download()``.
    :param headers:  forwarded to ``tools.file.download()``.
    :param strip_root: forwarded to ``tools.file.unzip()``.
    :param extract_filter: forwarded to ``tools.file.unzip()``.
    :param excludes: forwarded to ``tools.file.unzip()``.
    """

    if not filename:  # deduce filename from the URL
        url_base = url[0] if isinstance(url, (list, tuple)) else url
        if "?" in url_base or "=" in url_base:
            raise ConanException("Cannot deduce file name from the url: '{}'. Use 'filename' "
                                 "parameter.".format(url_base))
        filename = os.path.basename(url_base)

    download(conanfile, url, filename, verify=verify,
             retry=retry, retry_wait=retry_wait, auth=auth, headers=headers,
             md5=md5, sha1=sha1, sha256=sha256)
    unzip(conanfile, filename, destination=destination, keep_permissions=keep_permissions,
          pattern=pattern, strip_root=strip_root, extract_filter=extract_filter,
          excludes=excludes)
    os.unlink(filename)


def ftp_download(conanfile, host, filename, login='', password='', secure=False):
    """
    Ftp download of a file. Retrieves a file from an FTP server.

    :param conanfile: The current recipe object. Always use ``self``.
    :param host: IP or host of the FTP server.
    :param filename: Path to the file to be downloaded.
    :param login: Authentication login.
    :param password: Authentication password.
    :param secure: Set to True to use FTP over TLS/SSL (FTPS). Defaults to False for regular FTP.
    """
    # TODO: Check if we want to join this method with download() one, based on ftp:// protocol
    # this has been requested by some users, but the signature is a bit divergent
    import ftplib
    ftp = None
    try:
        if secure:
            ftp = ftplib.FTP_TLS(host)
            ftp.prot_p()
        else:
            ftp = ftplib.FTP(host)
        ftp.login(login, password)
        filepath, filename = os.path.split(filename)
        if filepath:
            ftp.cwd(filepath)
        with open(filename, 'wb') as f:
            ftp.retrbinary('RETR ' + filename, f.write)
    except Exception as e:
        try:
            os.unlink(filename)
        except OSError:
            pass
        raise ConanException("Error in FTP download from %s\n%s" % (host, str(e)))
    finally:
        if ftp:
            ftp.quit()


def download(conanfile, url, filename, verify=True, retry=None, retry_wait=None,
             auth=None, headers=None, md5=None, sha1=None, sha256=None):
    """
    Retrieves a file from a given URL into a file with a given filename. It uses certificates from
    a list of known verifiers for https downloads, but this can be optionally disabled.

    You can pass hash checking parameters: ``md5``, ``sha1``, ``sha256``. All the specified
    algorithms will be checked. If any of them doesn’t match, the downloaded file will be removed
    and it will raise a ``ConanException``.

    :param conanfile: The current recipe object. Always use ``self``.
    :param url: URL to download. It can be a list, which only the first one will be downloaded, and
                the follow URLs will be used as mirror in case of download error.  Files accessible
                in the local filesystem can be referenced with a URL starting with ``file:///``
                followed by an absolute path to a file (where the third ``/`` implies ``localhost``).
    :param filename: Name of the file to be created in the local storage
    :param verify: When False, disables https certificate validation
    :param retry: Number of retries in case of failure. Default is overridden by
           "tools.files.download:retry" conf
    :param retry_wait: Seconds to wait between download attempts. Default is overriden by
           "tools.files.download:retry_wait" conf.
    :param auth: A tuple of user and password to use HTTPBasic authentication
    :param headers: A dictionary with additional headers
    :param md5: MD5 hash code to check the downloaded file
    :param sha1: SHA-1 hash code to check the downloaded file
    :param sha256: SHA-256 hash code to check the downloaded file
    """
    config = conanfile.conf

    retry = retry if retry is not None else 2
    retry = config.get("tools.files.download:retry", check_type=int, default=retry)
    retry_wait = retry_wait if retry_wait is not None else 5
    retry_wait = config.get("tools.files.download:retry_wait", check_type=int, default=retry_wait)
    verify = config.get("tools.files.download:verify", check_type=bool, default=verify)

    filename = os.path.abspath(filename)
    downloader = SourcesCachingDownloader(conanfile)
    downloader.download(url, filename, retry, retry_wait, verify, auth, headers, md5, sha1, sha256)


def rename(conanfile, src, dst):
    """
    Utility functions to rename a file or folder src to dst with retrying. ``os.rename()``
    frequently raises “Access is denied” exception on Windows.
    This function renames file or folder using robocopy to avoid the exception on Windows.



    :param conanfile: The current recipe object. Always use ``self``.
    :param src: Path to be renamed.
    :param dst: Path to be renamed to.
    """

    # FIXME: This function has been copied from legacy. Needs to fix: which()
    # call and wrap subprocess call.
    if os.path.exists(dst):
        raise ConanException("rename {} to {} failed, dst exists.".format(src, dst))

    if platform.system() == "Windows" and which("robocopy") and os.path.isdir(src):
        # /move Moves files and directories, and deletes them from the source after they are copied.
        # /e Copies subdirectories. Note that this option includes empty directories.
        # /ndl Specifies that directory names are not to be logged.
        # /nfl Specifies that file names are not to be logged.
        process = subprocess.Popen(["robocopy", "/move", "/e", "/ndl", "/nfl", src, dst],
                                   stdout=subprocess.PIPE)
        process.communicate()
        if process.returncode > 7:  # https://ss64.com/nt/robocopy-exit.html
            raise ConanException("rename {} to {} failed.".format(src, dst))
    else:
        try:
            os.rename(src, dst)
        except Exception as err:
            raise ConanException("rename {} to {} failed: {}".format(src, dst, err))


@contextmanager
def chdir(conanfile, newdir):
    """
    This is a context manager that allows to temporary change the current directory in your conanfile

    :param conanfile: The current recipe object. Always use ``self``.
    :param newdir: Directory path name to change the current directory.

    """
    old_path = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(old_path)


def chmod(conanfile, path:str, read:Optional[bool]=None, write:Optional[bool]=None, execute:Optional[bool]=None, recursive:bool=False):
    """Change file or directory permissions cross-platform.

    .. versionadded:: 2.15

    This function is a simple wrapper around the chmod Unix command, but it is cross-platform supported.
    It is indicated to use it instead of os.stat + os.chmod, as it only changes the permissions of the
    directory or file for the owner and avoids issues with the umask.
    On Windows is limited to changing write permission only.

    Parameters
    ----------
    conanfile : object
        The current recipe object. Always use ``self``.
    path : str
        Path to the file or directory whose permissions will be changed.
    read : bool, optional
        If ``True``, the file or directory will be given read permissions for owner user.
        If ``False``, the read permission will be removed.
        If ``None``, the read permission will be left unchanged.
        Defaults to None.
    write : bool, optional
        If ``True``, the file or directory will be given write permissions for owner user.
        If ``False``, the write permission will be removed.
        If ``None``, the file or directory will not be changed.
        Defaults to None.
    execute : bool, optional
        If ``True``, the file or directory will be given execute permissions for owner user.
        If ``False``, the execution permission will be removed.
        If ``None``, the file or directory will not be changed.
        Defaults to None.
    recursive : bool
        If ``True``, the permissions will be applied recursively to all files and directories
        inside the specified directory. If ``False``, only the specified file or directory will
        be changed. Defaults to False.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python
        :caption: Add execution permission to a packaged bash script

        from conan.tools.files import chmod
        chmod(self, os.path.join(self.package_folder, "bin", "script.sh"), execute=True)
    """
    if read is None and write is None and execute is None:
        raise ConanException("Could not change permission: At least one of the permissions should be set.")

    if not os.path.exists(path):
        raise ConanException(f"Could not change permission: Path \"{path}\" does not exist.")

    def _change_permission(it_path:str):
        mode = os.stat(it_path).st_mode
        permissions = [
            (read, stat.S_IRUSR),
            (write, stat.S_IWUSR),
            (execute, stat.S_IXUSR)
        ]
        for enabled, mask in permissions:
            if enabled is None:
                continue
            elif enabled:
                mode |= mask
            else:
                mode &= ~mask
        os.chmod(it_path, mode)

    if recursive:
        for root, _, files in os.walk(path):
            for file in files:
                _change_permission(os.path.join(root, file))
    else:
        _change_permission(path)


def unzip(conanfile, filename, destination=".", keep_permissions=False, pattern=None,
          strip_root=False, extract_filter=None, excludes=None):
    """
    Extract different compressed formats

    :param conanfile: The current recipe object. Always use ``self``.
    :param filename: Path to the compressed file.
    :param destination: (Optional, Defaulted to ``.``) Destination folder (or file for .gz files)
    :param keep_permissions: (Optional, Defaulted to ``False``) Keep the zip permissions.
           WARNING: Can be dangerous if the zip was not created in a NIX system, the bits could
           produce undefined permission schema. Use this option only if you are sure that the zip
           was created correctly.
    :param pattern: (Optional, Defaulted to ``None``) Extract only paths matching the pattern.
           This should be a Unix shell-style wildcard, see fnmatch documentation for more details.
    :param strip_root: (Optional, Defaulted to False) If True, and all the unzipped contents are
           in a single folder it will flat the folder moving all the contents to the parent folder.
    :param extract_filter: (Optional, defaulted to None). When extracting a tar file,
           use the tar extracting filters define by Python in
           https://docs.python.org/3/library/tarfile.html
    :param excludes: (Optional, defaulted to None). When extracting a file,
           exclude paths matching any of the patterns. This should be a Unix shell-style wildcard,
           see fnmatch documentation for more details.
    """

    output = conanfile.output
    extract_filter = conanfile.conf.get("tools.files.unzip:filter") or extract_filter
    output.info(f"Uncompressing {filename} to {destination}")
    if (filename.endswith(".tar.gz") or filename.endswith(".tgz") or
            filename.endswith(".tbz2") or filename.endswith(".tar.bz2") or
            filename.endswith(".tar")):
        return untargz(filename, destination, pattern, strip_root, extract_filter,
                       excludes=excludes)
    if filename.endswith(".gz"):
        target_name = filename[:-3] if destination == "." else destination
        target_dir = os.path.dirname(target_name)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with gzip.open(filename, 'rb') as fin:
            with open(target_name, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        return
    if filename.endswith(".tar.xz") or filename.endswith(".txz"):
        return untargz(filename, destination, pattern, strip_root, extract_filter,
                       excludes=excludes)

    import zipfile
    full_path = os.path.normpath(os.path.join(os.getcwd(), destination))

    with FileProgress(filename, msg="Unzipping", mode="r") as file, zipfile.ZipFile(file) as z:
        zip_info = z.infolist()
        if pattern:
            zip_info = [zi for zi in zip_info if fnmatch(zi.filename, pattern)]
        if excludes:
            zip_info = [zi for zi in zip_info
                        if not any(fnmatch(zi.filename, pat) for pat in excludes)]
        if strip_root:
            names = [zi.filename.replace("\\", "/") for zi in zip_info]
            common_folder = os.path.commonprefix(names).split("/", 1)[0]
            if not common_folder and len(names) > 1:
                raise ConanException("The zip file contains more than 1 folder in the root")
            if len(names) == 1 and len(names[0].split("/", 1)) == 1:
                raise ConanException("The zip file contains a file in the root")
            # Remove the directory entry if present
            # Note: The "zip" format contains the "/" at the end if it is a directory
            zip_info = [m for m in zip_info if m.filename != (common_folder + "/")]
            for member in zip_info:
                name = member.filename.replace("\\", "/")
                member.filename = name.split("/", 1)[1]

        uncompress_size = sum((file_.file_size for file_ in zip_info))
        if uncompress_size > 100000:
            output.info("Unzipping %s, this can take a while" % human_size(uncompress_size))
        else:
            output.info("Unzipping %s" % human_size(uncompress_size))
        extracted_size = 0

        if platform.system() == "Windows":
            for file_ in zip_info:
                extracted_size += file_.file_size
                try:
                    z.extract(file_, full_path)
                except Exception as e:
                    output.error(f"Error extract {file_.filename}\n{str(e)}", error_type="exception")
        else:  # duplicated for, to avoid a platform check for each zipped file
            for file_ in zip_info:
                extracted_size += file_.file_size
                try:
                    z.extract(file_, full_path)
                    if keep_permissions:
                        # Could be dangerous if the ZIP has been created in a non nix system
                        # https://bugs.python.org/issue15795
                        perm = file_.external_attr >> 16 & 0xFFF
                        os.chmod(os.path.join(full_path, file_.filename), perm)
                except Exception as e:
                    output.error(f"Error extract {file_.filename}\n{str(e)}", error_type="exception")
        output.writeln("")


def untargz(filename, destination=".", pattern=None, strip_root=False, extract_filter=None,
            excludes=None):
    # NOT EXPOSED at `conan.tools.files` but used in tests
    import tarfile
    with tarfile.TarFile.open(filename, mode='r:*') as tarredgzippedFile:
        f = getattr(tarfile, f"{extract_filter}_filter", None) if extract_filter else None
        tarredgzippedFile.extraction_filter = f or (lambda member_, _: member_)
        # https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
        # File I/O functions in the Windows API convert "/" to "\" as part of converting
        # the name to an NT-style name, except when using the "\\?\" prefix
        using_long_path_prefix = destination.startswith("\\\\?\\")
        if not pattern and not excludes and not strip_root and not using_long_path_prefix:
            tarredgzippedFile.extractall(destination)
        else:
            common_folder = None
            members = []
            for member in tarredgzippedFile:
                if pattern and not fnmatch(member.name, pattern):
                    continue  # Skip files that don’t match the pattern
                if excludes and any(fnmatch(member.name, pat) for pat in excludes):
                    continue  # Skip files that match the excludes

                if strip_root:
                    name = member.name.replace("\\", "/")
                    if not common_folder:
                        splits = name.split("/", 1)
                        # First case for a plain folder in the root
                        if member.isdir() or len(splits) > 1:
                            common_folder = splits[0]  # Find the root folder
                        else:
                            raise ConanException("Can't untar a tgz containing files in the root with strip_root enabled")
                    if not name.startswith(common_folder):
                        raise ConanException("The tgz file contains more than 1 folder in the root")
                    # Adjust the member's name for extraction
                    member.name = name[len(common_folder) + 1:]
                    member.path = member.name
                    if member.linkpath and member.linkpath.startswith(common_folder):
                        # https://github.com/conan-io/conan/issues/11065
                        member.linkpath = member.linkpath[len(common_folder) + 1:].replace("\\", "/")
                        member.linkname = member.linkpath
                if using_long_path_prefix:
                    member.name = member.name.replace("/", "\\")
                # Let's gather each member
                members.append(member)
            tarredgzippedFile.extractall(destination, members=members)


def check_sha1(conanfile, file_path, signature):
    """
    Check that the specified ``sha1`` of the ``file_path`` matches with signature.
    If doesn’t match it will raise a ``ConanException``.

    :param conanfile: Conanfile object.
    :param file_path: Path of the file to check.
    :param signature: Expected sha1sum
    """
    check_with_algorithm_sum("sha1", file_path, signature)


def check_md5(conanfile, file_path, signature):
    """
    Check that the specified ``md5sum`` of the ``file_path`` matches with ``signature``.
    If doesn’t match it will raise a ``ConanException``.

    :param conanfile: The current recipe object. Always use ``self``.
    :param file_path: Path of the file to check.
    :param signature: Expected md5sum.
    """
    check_with_algorithm_sum("md5", file_path, signature)


def check_sha256(conanfile, file_path, signature):
    """
    Check that the specified ``sha256`` of the ``file_path`` matches with signature.
    If doesn’t match it will raise a ``ConanException``.

    :param conanfile: Conanfile object.
    :param file_path: Path of the file to check.
    :param signature: Expected sha256sum
    """
    check_with_algorithm_sum("sha256", file_path, signature)


def replace_in_file(conanfile, file_path, search, replace, strict=True, encoding="utf-8"):
    """
    Replace a string ``search`` in the contents of the file ``file_path`` with the string replace.

    :param conanfile: The current recipe object. Always use ``self``.
    :param file_path: File path of the file to perform the replacing.
    :param search: String you want to be replaced.
    :param replace: String to replace the searched string.
    :param strict: (Optional, Defaulted to ``True``) If ``True``, it raises an error if the searched
           string is not found, so nothing is actually replaced.
    :param encoding: (Optional, Defaulted to utf-8): Specifies the input and output files text
           encoding.
    :return: ``True`` if the pattern was found, ``False`` otherwise if `strict` is ``False``.
    """
    output = conanfile.output
    content = load(conanfile, file_path, encoding=encoding)
    if -1 == content.find(search):
        message = "replace_in_file didn't find pattern '%s' in '%s' file." % (search, file_path)
        if strict:
            raise ConanException(message)
        else:
            output.warning(message)
            return False
    content = content.replace(search, replace)
    save(conanfile, file_path, content, encoding=encoding)
    return True


def collect_libs(conanfile, folder=None):
    """
    Returns a sorted list of library names from the libraries (files with extensions *.so*, *.lib*,
    *.a* and *.dylib*) located inside the ``conanfile.cpp_info.libdirs`` (by default) or the
    **folder** directory relative to the package folder. Useful to collect not inter-dependent
    libraries or with complex names like ``libmylib-x86-debug-en.lib``.

    For UNIX libraries staring with **lib**, like *libmath.a*, this tool will collect the library
    name **math**.

    :param conanfile: The current recipe object. Always use ``self``.
    :param folder: (Optional, Defaulted to ``None``): String indicating the subfolder name inside
           ``conanfile.package_folder`` where the library files are.
    :return: A list with the library names
    """
    if not conanfile.package_folder:
        return []
    if folder:
        lib_folders = [os.path.join(conanfile.package_folder, folder)]
    else:
        lib_folders = [os.path.join(conanfile.package_folder, folder)
                       for folder in conanfile.cpp_info.libdirs]

    ref_libs = {}
    for lib_folder in lib_folders:
        if not os.path.exists(lib_folder):
            conanfile.output.warning("Lib folder doesn't exist, can't collect libraries: "
                                     "{0}".format(lib_folder))
            continue
        # In case of symlinks, only keep shortest file name in the same "group"
        files = os.listdir(lib_folder)
        for f in files:
            name, ext = os.path.splitext(f)
            if ext in (".so", ".lib", ".a", ".dylib", ".bc"):
                real_lib = os.path.basename(os.path.realpath(os.path.join(lib_folder, f)))
                if real_lib not in ref_libs or len(f) < len(ref_libs[real_lib]):
                    ref_libs[real_lib] = f

    result = []
    for f in ref_libs.values():
        name, ext = os.path.splitext(f)
        if ext != ".lib" and name.startswith("lib"):
            name = name[3:]
        result.append(name)
    result.sort()
    return result


def move_folder_contents(conanfile, src_folder, dst_folder):
    """ replaces the dst_folder contents with the contents of the src_folder, which can be a
    child folder of dst_folder. This is used in the SCM monorepo flow, when it is necessary
    to use one subproject subfolder to replace the whole cloned git repo
    /base-folder                       /base-folder
        /pkg  (src folder)                 /other/<otherfiles>
          /other/<otherfiles>              /pkg/<pkgfiles>
          /pkg/<pkgfiles>                  <files>
          <files>
        /siblings
        <siblingsfiles>
    """
    # Remove potential "siblings" folders not wanted
    src_folder_name = os.path.basename(src_folder)
    for f in os.listdir(dst_folder):
        if f != src_folder_name:  # FIXME: Only works for 1st level subfolder
            dst = os.path.join(dst_folder, f)
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                _internal_rmdir(dst)

    # Move all the contents
    for f in os.listdir(src_folder):
        src = os.path.join(src_folder, f)
        dst = os.path.join(dst_folder, f)
        if not os.path.exists(dst):
            shutil.move(src, dst_folder)
        else:
            for sub_src in os.listdir(src):
                shutil.move(os.path.join(src, sub_src), dst)
            _internal_rmdir(src)
    try:
        os.rmdir(src_folder)
    except OSError:
        pass
