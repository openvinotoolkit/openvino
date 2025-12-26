import errno
import hashlib
import os
import platform
import shutil
import stat
import sys
import tarfile
import time

from contextlib import contextmanager

from conan.errors import ConanException

_DIRTY_FOLDER = ".dirty"


def set_dirty(folder):
    dirty_file = os.path.normpath(folder) + _DIRTY_FOLDER
    assert not os.path.exists(dirty_file), "Folder '{}' is already dirty".format(folder)
    save(dirty_file, "")


def clean_dirty(folder):
    dirty_file = os.path.normpath(folder) + _DIRTY_FOLDER
    os.remove(dirty_file)


def is_dirty(folder):
    dirty_file = os.path.normpath(folder) + _DIRTY_FOLDER
    return os.path.exists(dirty_file)


def remove_if_dirty(item):
    # TODO: Apply to other places this pattern is common
    if is_dirty(item):
        if os.path.exists(item):
            # To avoid circular import in conan_server
            from conan.api.output import ConanOutput
            ConanOutput().warning(f"{item} is dirty, removing it")
            if os.path.isfile(item):
                os.remove(item)
            else:
                rmdir(item)
        clean_dirty(item)
        return True
    return False


@contextmanager
def set_dirty_context_manager(folder):
    set_dirty(folder)
    yield
    clean_dirty(folder)


@contextmanager
def chdir(newdir):
    old_path = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(old_path)


def md5(content):
    try:
        md5alg = hashlib.md5()
    except ValueError:  # FIPS error https://github.com/conan-io/conan/issues/7800
        md5alg = hashlib.md5(usedforsecurity=False)
    if isinstance(content, bytes):
        tmp = content
    else:
        tmp = content.encode("utf-8")
    md5alg.update(tmp)
    return md5alg.hexdigest()


def md5sum(file_path):
    return _generic_algorithm_sum(file_path, "md5")


def sha1sum(file_path):
    return _generic_algorithm_sum(file_path, "sha1")


def sha256sum(file_path):
    return _generic_algorithm_sum(file_path, "sha256")


def _generic_algorithm_sum(file_path, algorithm_name):

    with open(file_path, 'rb') as fh:
        try:
            m = hashlib.new(algorithm_name)
        except ValueError:  # FIPS error https://github.com/conan-io/conan/issues/7800
            m = hashlib.new(algorithm_name, usedforsecurity=False)
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
        return m.hexdigest()


def check_with_algorithm_sum(algorithm_name, file_path, signature):
    real_signature = _generic_algorithm_sum(file_path, algorithm_name)
    if real_signature != signature.lower():
        raise ConanException("%s signature failed for '%s' file. \n"
                             " Provided signature: %s  \n"
                             " Computed signature: %s" % (algorithm_name,
                                                          os.path.basename(file_path),
                                                          signature,
                                                          real_signature))


def save(path, content, encoding="utf-8"):
    """
    Saves a file with given content
    Params:
        path: path to write file to
        content: contents to save in the file
        encoding: target file text encoding
    """

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(content)


def save_files(path, files, encoding="utf-8"):
    for name, content in files.items():
        save(os.path.join(path, name), content, encoding=encoding)


def load(path, encoding="utf-8"):
    """ Loads a file content """
    with open(path, 'r', encoding=encoding, newline="") as handle:
        tmp = handle.read()
    return tmp


def load_user_encoded(path):
    """ Exclusive for user side read-only files:
     - conanfile.txt
     - profile files
     """
    with open(path, 'rb') as handle:
        text = handle.read()

    import codecs
    encodings = {codecs.BOM_UTF8: "utf_8_sig",
                 codecs.BOM_UTF32_BE: "utf_32_be",
                 codecs.BOM_UTF32_LE: "utf_32_le",
                 codecs.BOM_UTF16_BE: "utf_16_be",
                 codecs.BOM_UTF16_LE: "utf_16_le",
                 b'\x2b\x2f\x76\x38': "utf_7",
                 b'\x2b\x2f\x76\x39': "utf_7",
                 b'\x2b\x2f\x76\x2b': "utf_7",
                 b'\x2b\x2f\x76\x2f': "utf_7",
                 b'\x2b\x2f\x76\x38\x2d': "utf_7"}
    for bom, encoding in encodings.items():
        if text.startswith(bom):
            return text[len(bom):].decode(encoding)

    for decoder in ["utf-8", "Windows-1252"]:
        try:
            return text.decode(decoder)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Unknown encoding of file: {path}\nIt is recommended to use utf-8 encoding")


def _change_permissions(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise OSError("Cannot change permissions for {}! Exception info: {}".format(path, exc_info))


if platform.system() == "Windows":
    def rmdir(path):
        if not os.path.isdir(path):
            return

        retries = 3
        delay = 0.5
        for i in range(retries):
            try:
                shutil.rmtree(path, onerror=_change_permissions)
                break
            except OSError as err:
                if i == retries - 1:
                    raise ConanException(f"Couldn't remove folder: {path}\n{str(err)}\n"
                                         "Folder might be busy or open. "
                                         "Close any app using it and retry.")
                time.sleep(delay)


    def renamedir(old_path, new_path):
        retries = 3
        delay = 0.5
        for i in range(retries):
            try:
                shutil.move(old_path, new_path)
                break
            except OSError as err:
                if i == retries - 1:
                    raise ConanException(f"Couldn't move folder: {old_path}->{new_path}\n"
                                         f"{str(err)}\n"
                                         "Folder might be busy or open. "
                                         "Close any app using it and retry.")
                time.sleep(delay)
else:
    def rmdir(path):
        if not os.path.isdir(path):
            return
        try:
            shutil.rmtree(path, onerror=_change_permissions)
        except OSError as err:
            raise ConanException(f"Couldn't remove folder: {path}\n{str(err)}\n"
                                 "Folder might be busy or open. "
                                 "Close any app using it and retry.")

    def renamedir(old_path, new_path):
        try:
            shutil.move(old_path, new_path)
        except OSError as err:
            raise ConanException(
                f"Couldn't move folder: {old_path}->{new_path}\n{str(err)}\n"
                "Folder might be busy or open. "
                "Close any app using it and retry.")


def remove(path):
    try:
        assert os.path.isfile(path)
        os.remove(path)
    except (IOError, OSError) as e:  # for py3, handle just PermissionError
        if e.errno == errno.EPERM or e.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU)
            os.remove(path)
            return
        raise


def mkdir(path):
    """Recursive mkdir, doesnt fail if already existing"""
    if os.path.exists(path):
        return
    os.makedirs(path)


def tar_extract(fileobj, destination_dir):
    the_tar = tarfile.open(fileobj=fileobj)
    # NOTE: The errorlevel=2 has been removed because it was failing in Win10, it didn't allow to
    # "could not change modification time", with time=0
    # the_tar.errorlevel = 2  # raise exception if any error
    the_tar.extraction_filter = (lambda member, path: member)  # fully_trusted, avoid Py3.14 break
    the_tar.extractall(path=destination_dir)
    the_tar.close()


def merge_directories(src, dst):
    from conan.tools.files import copy
    copy(None, pattern="*", src=src, dst=dst)


def gather_files(folder):
    file_dict = {}
    symlinked_folders = {}
    for root, dirs, files in os.walk(folder):
        if root != folder and not dirs and not files:  # empty folder
            rel_path = root[len(folder) + 1:].replace("\\", "/")
            symlinked_folders[rel_path] = root
            continue
        for d in dirs:
            abs_path = os.path.join(root, d)
            if os.path.islink(abs_path):
                rel_path = abs_path[len(folder) + 1:].replace("\\", "/")
                symlinked_folders[rel_path] = abs_path
        for f in files:
            if f == ".DS_Store":
                continue
            abs_path = os.path.join(root, f)
            rel_path = abs_path[len(folder) + 1:].replace("\\", "/")
            file_dict[rel_path] = abs_path

    return file_dict, symlinked_folders


def human_size(size_bytes):
    """
    format a size in bytes into a 'human' file size, e.g. B, KB, MB, GB, TB, PB
    Note that bytes will be reported in whole numbers but KB and above will have
    greater precision.  e.g. 43 B, 443 KB, 4.3 MB, 4.43 GB, etc
    """
    unit_size = 1000.0
    suffixes_table = [('B', 0), ('KB', 1), ('MB', 1), ('GB', 2), ('TB', 2), ('PB', 2)]

    num = float(size_bytes)
    the_precision = None
    the_suffix = None
    for suffix, precision in suffixes_table:
        the_precision = precision
        the_suffix = suffix
        if num < unit_size:
            break
        num /= unit_size

    if the_precision == 0:
        formatted_size = "%d" % num
    else:
        formatted_size = str(round(num, ndigits=the_precision))

    return "%s%s" % (formatted_size, the_suffix)


# FIXME: completely remove disutils once we don't support <3.8 any more
def copytree_compat(source_folder, dest_folder):
    if sys.version_info >= (3, 8):
        shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
    else:
        from distutils.dir_util import copy_tree
        copy_tree(source_folder, dest_folder)
