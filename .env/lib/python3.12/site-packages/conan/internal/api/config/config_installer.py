import os
import shutil
import fnmatch
import zipfile

from urllib.parse import urlparse, urlsplit
from contextlib import contextmanager

from conan.api.output import ConanOutput
from conan.internal.paths import find_file_walk_up
from conan.internal.rest.file_downloader import FileDownloader
from conan.errors import ConanException
from conan.internal.util.files import mkdir, rmdir, remove, chdir
from conan.internal.util.runners import detect_runner
from conan.tools.files.files import untargz


class _ConanIgnoreMatcher:
    def __init__(self, conanignore_path, ignore=None):
        self._ignored_entries = {".conanignore"}
        self._included_entries = set()
        if ignore:
            self._ignored_entries.update(ignore)
        if conanignore_path is None or not os.path.exists(conanignore_path):
            return
        with open(conanignore_path, 'r') as conanignore:
            for line in conanignore:
                line_content = line.split("#", maxsplit=1)[0].strip()
                if line_content:
                    if line_content.startswith("!"):
                        self._included_entries.add(line_content[1:])
                    else:
                        self._ignored_entries.add(line_content)

    def matches(self, path):
        """Returns whether the path should be ignored

        It's ignored if:
         - The path does not match any of the included entries
         - And the path matches any of the ignored entries

        In any other, the path is not ignored"""
        for include_entry in self._included_entries:
            if fnmatch.fnmatch(path, include_entry):
                return False
        for ignore_entry in self._ignored_entries:
            if fnmatch.fnmatch(path, ignore_entry):
                return True
        return False


def _hide_password(resource):
    """
    Hide password from url/file path

    :param resource: string with url or file path
    :return: resource with hidden password if present
    """
    password = urlparse(resource).password
    return resource.replace(password, "<hidden>") if password else resource


@contextmanager
def tmp_config_install_folder(cache_folder):
    tmp_folder = os.path.join(cache_folder, "tmp_config_install")
    # necessary for Mac OSX, where the temp folders in /var/ are symlinks to /private/var/
    tmp_folder = os.path.abspath(tmp_folder)
    rmdir(tmp_folder)
    mkdir(tmp_folder)
    try:
        yield tmp_folder
    finally:
        rmdir(tmp_folder)


def _process_git_repo(config, cache_folder):
    output = ConanOutput()
    output.info("Trying to clone repo: %s" % config.uri)
    with tmp_config_install_folder(cache_folder) as tmp_folder:
        with chdir(tmp_folder):
            args = config.args or ""
            ret, out = detect_runner('git clone "{}" . {}'.format(config.uri, args))
            if ret != 0:
                raise ConanException("Can't clone repo: {}".format(out))
            output.info("Repo cloned!")
        _process_folder(config, tmp_folder, cache_folder)


def _process_zip_file(config, zippath, cache_folder, tmp_folder, first_remove=False):
    # First, unzip. This is repeated with the tools.unzip, but better do not mess
    # Same list as below
    tgz_exts = (".tar.gz", ".tgz", ".tbz2", ".tar.bz2", ".tar", ".tar.xz", ".txz")
    if any(zippath.endswith(e) for e in tgz_exts):
        untargz(zippath, tmp_folder)
    else:
        full_path = os.path.normpath(os.path.join(os.getcwd(), tmp_folder))
        with zipfile.ZipFile(zippath, "r") as z:
            zip_info = z.infolist()
            extracted_size = 0
            for file_ in zip_info:
                extracted_size += file_.file_size
                z.extract(file_, full_path)

    if first_remove:
        os.unlink(zippath)
    _process_folder(config, tmp_folder, cache_folder)


def _filecopy(src, filename, dst):
    # https://github.com/conan-io/conan/issues/6556
    # This is just a local convenience for "conan config install", using copyfile to avoid
    # copying with permissions that later cause bugs
    src = os.path.join(src, filename)
    dst = os.path.join(dst, filename)
    # Clear the destination file
    if os.path.exists(dst):
        if os.path.isdir(dst):  # dst was a directory and now src is a file
            rmdir(dst)
        else:
            remove(dst)
    shutil.copyfile(src, dst)


def _process_file(directory, filename, config, cache_folder, folder):
    output = ConanOutput()
    if filename == "settings.yml":
        output.info("Installing settings.yml")
        _filecopy(directory, filename, cache_folder)
    elif filename == "remotes.json":
        output.info("Defining remotes from remotes.json")
        _filecopy(directory, filename, cache_folder)
    else:
        relpath = os.path.relpath(directory, folder)
        if config.target_folder:
            target_folder = os.path.join(cache_folder, config.target_folder, relpath)
        else:
            target_folder = os.path.join(cache_folder, relpath)

        if os.path.isfile(target_folder):  # Existed as a file and now should be a folder
            remove(target_folder)

        mkdir(target_folder)
        output.info("Copying file %s to %s" % (filename, target_folder))
        _filecopy(directory, filename, target_folder)


def _process_folder(config, folder, cache_folder, ignore=None):
    if not os.path.isdir(folder):
        raise ConanException("No such directory: '%s'" % str(folder))
    original_folder = folder
    if config.source_folder:
        folder = os.path.join(folder, config.source_folder)
    conanignore_path = find_file_walk_up(folder, ".conanignore", end=original_folder)
    conanignore = _ConanIgnoreMatcher(conanignore_path, ignore)
    for root, dirs, files in os.walk(folder):
        # .git is always ignored by default, even if not present in .conanignore
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), folder)
            if not conanignore.matches(rel_path):
                _process_file(root, f, config, cache_folder, folder)


def _process_download(config, cache_folder, requester):
    output = ConanOutput()
    with tmp_config_install_folder(cache_folder) as tmp_folder:
        output.info("Trying to download  %s" % _hide_password(config.uri))
        path = urlsplit(config.uri).path
        filename = os.path.basename(path)
        zippath = os.path.join(tmp_folder, filename)
        try:
            downloader = FileDownloader(requester=requester, source_credentials=True)
            downloader.download(url=config.uri, file_path=zippath, verify_ssl=config.verify_ssl,
                                retry=1)
            _process_zip_file(config, zippath, cache_folder, tmp_folder, first_remove=True)
        except Exception as e:
            raise ConanException("Error while installing config from %s\n%s" % (config.uri, str(e)))


class _ConfigOrigin(object):
    def __init__(self, uri, config_type, verify_ssl, args, source_folder, target_folder):
        if config_type:
            self.type = config_type
        else:
            if uri.endswith(".git"):
                self.type = "git"
            elif os.path.isdir(uri):
                self.type = "dir"
            elif os.path.isfile(uri):
                self.type = "file"
            elif uri.startswith("http"):
                self.type = "url"
            else:
                raise ConanException("Unable to deduce type config install: %s" % uri)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.args = args
        self.verify_ssl = verify_ssl
        if os.path.exists(uri):
            uri = os.path.abspath(uri)
        self.uri = uri


def _is_compressed_file(filename):
    open(filename, "r")  # Check if the file exist and can be opened
    import zipfile
    if zipfile.is_zipfile(filename):
        return True
    tgz_exts = (".tar.gz", ".tgz", ".tbz2", ".tar.bz2", ".tar", ".tar.xz", ".txz")
    return any(filename.endswith(e) for e in tgz_exts)


def configuration_install(cache_folder, requester, uri, verify_ssl, config_type=None,
                          args=None, source_folder=None, target_folder=None, ignore=None):
    config = _ConfigOrigin(uri, config_type, verify_ssl, args, source_folder, target_folder)
    try:
        if config.type == "git":
            _process_git_repo(config, cache_folder)
        elif config.type == "dir":
            _process_folder(config, config.uri, cache_folder, ignore)
        elif config.type == "file":
            if _is_compressed_file(config.uri):
                with tmp_config_install_folder(cache_folder) as tmp_folder:
                    _process_zip_file(config, config.uri, cache_folder, tmp_folder)
            else:
                dirname, filename = os.path.split(config.uri)
                _process_file(dirname, filename, config, cache_folder, dirname)
        elif config.type == "url":
            _process_download(config, cache_folder, requester=requester)
        else:
            raise ConanException("Unable to process config install: %s" % config.uri)
    except Exception as e:
        raise ConanException("Failed conan config install: %s" % str(e))
