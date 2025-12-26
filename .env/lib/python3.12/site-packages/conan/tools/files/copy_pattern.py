import filecmp
import fnmatch
import os
import shutil

from conan.errors import ConanException
from conan.internal.util.files import mkdir


def copy(conanfile, pattern, src, dst, keep_path=True, excludes=None,
         ignore_case=True, overwrite_equal=False):
    """
    Copy the files matching the pattern (fnmatch) at the src folder to a dst folder.

    :param conanfile: The current recipe object. Always use ``self``.
    :param pattern: (Required) An fnmatch file pattern of the files that should be copied.
           It must not start with ``..`` relative path or an exception will be raised.
    :param src: (Required) Source folder in which those files will be searched. This folder
           will be stripped from the dst parameter. E.g., lib/Debug/x86.
    :param dst: (Required) Destination local folder. It must be different from src value or an
           exception will be raised.
    :param keep_path: (Optional, defaulted to ``True``) Means if you want to keep the relative
           path when you copy the files from the src folder to the dst one.
    :param excludes: (Optional, defaulted to ``None``) A tuple/list of fnmatch patterns or even a
           single one to be excluded from the copy.
    :param ignore_case: (Optional, defaulted to ``True``) If enabled, it will do a
           case-insensitive pattern matching. will do a case-insensitive pattern matching when
           ``True``
    :param overwrite_equal: (Optional, default ``False``). If the file to be copied already exists
           in the destination folder, only really copy it if it seems different (different size,
           different modification time)
    :return: list of copied files
    """
    if src == dst:
        raise ConanException("copy() 'src' and 'dst' arguments must have different values")
    if pattern.startswith(".."):
        raise ConanException("copy() it is not possible to use relative patterns starting with '..'")
    if src is None:
        raise ConanException("copy() received 'src=None' argument")

    # This is necessary to add the trailing / so it is not reported as symlink
    src = os.path.join(src, "")
    excluded_folder = dst
    files_to_copy, files_symlinked_to_folders = _filter_files(src, pattern, excludes, ignore_case,
                                                              excluded_folder)

    copied_files = _copy_files(files_to_copy, src, dst, keep_path, overwrite_equal)
    copied_files.extend(_copy_files_symlinked_to_folders(files_symlinked_to_folders, src, dst))
    if conanfile:  # Some usages still pass None
        copied = '\n    '.join(files_to_copy)
        conanfile.output.debug(f"copy(pattern={pattern}) copied {len(copied_files)} files\n"
                               f"  from {src}\n"
                               f"  to {dst}\n"
                               f"  Files:\n    {copied}")
    return copied_files


def _filter_files(src, pattern, excludes, ignore_case, excluded_folder):
    """ return a list of the files matching the patterns
    The list will be relative path names wrt to the root src folder
    """
    filenames = []
    files_symlinked_to_folders = []

    if excludes:
        if not isinstance(excludes, (tuple, list)):
            excludes = (excludes, )
        if ignore_case:
            excludes = [e.lower() for e in excludes]
    else:
        excludes = []

    for root, subfolders, files in os.walk(src):
        if root == excluded_folder:
            subfolders[:] = []
            continue

        # Check if any of the subfolders is a symlink
        for subfolder in subfolders:
            relative_path = os.path.relpath(os.path.join(root, subfolder), src)
            if os.path.islink(os.path.join(root, subfolder)):
                if fnmatch.fnmatch(os.path.normpath(relative_path.lower()), pattern):
                    files_symlinked_to_folders.append(relative_path)

        relative_path = os.path.relpath(root, src)
        compare_relative_path = relative_path.lower() if ignore_case else relative_path
        # Don't try to exclude the start folder, it conflicts with excluding names starting with dots
        if not compare_relative_path == ".":
            for exclude in excludes:
                if fnmatch.fnmatch(compare_relative_path, exclude):
                    subfolders[:] = []
                    files = []
                    break
        for f in files:
            relative_name = os.path.normpath(os.path.join(relative_path, f))
            filenames.append(relative_name)

    if ignore_case:
        pattern = pattern.lower()
        files_to_copy = [n for n in filenames if fnmatch.fnmatch(os.path.normpath(n.lower()),
                                                                 pattern)]
    else:
        files_to_copy = [n for n in filenames if fnmatch.fnmatchcase(os.path.normpath(n),
                                                                     pattern)]

    for exclude in excludes:
        if ignore_case:
            files_to_copy = [f for f in files_to_copy if not fnmatch.fnmatch(f.lower(), exclude)]
            files_symlinked_to_folders =\
                [f for f in files_symlinked_to_folders if not fnmatch.fnmatch(f.lower(), exclude)]
        else:
            files_to_copy = [f for f in files_to_copy if not fnmatch.fnmatchcase(f, exclude)]
            files_symlinked_to_folders =\
                [f for f in files_symlinked_to_folders if not fnmatch.fnmatchcase(f, exclude)]

    return files_to_copy, files_symlinked_to_folders


def _copy_files(files, src, dst, keep_path, overwrite_equal):
    """ executes a multiple file copy from [(src_file, dst_file), (..)]
    managing symlinks if necessary
    """
    copied_files = []
    for filename in files:
        abs_src_name = os.path.join(src, filename)
        filename = filename if keep_path else os.path.basename(filename)
        abs_dst_name = os.path.normpath(os.path.join(dst, filename))
        parent_folder = os.path.dirname(abs_dst_name)
        if parent_folder:  # There are cases where this folder will be empty for relative paths
            os.makedirs(parent_folder, exist_ok=True)

        if os.path.islink(abs_src_name):
            linkto = os.readlink(abs_src_name)
            try:
                os.remove(abs_dst_name)
            except OSError:
                pass
            os.symlink(linkto, abs_dst_name)
        else:
            # Avoid the copy if the file exists and has the exact same signature (size + mod time)
            if overwrite_equal or not os.path.exists(abs_dst_name) \
                    or not filecmp.cmp(abs_src_name, abs_dst_name):
                shutil.copy2(abs_src_name, abs_dst_name)
        copied_files.append(abs_dst_name)
    return copied_files


def _copy_files_symlinked_to_folders(files_symlinked_to_folders, src, dst):
    """Copy the files that are symlinks to folders from src to dst.
       The files are already filtered with the specified pattern"""
    copied_files = []
    for relative_path in files_symlinked_to_folders:
        abs_path = os.path.join(src, relative_path)
        symlink_path = os.path.join(dst, relative_path)
        # We create the same symlink in dst, no matter if it is absolute or relative
        link_dst = os.readlink(abs_path)  # This could be perfectly broken

        # Create the parent directory that will contain the symlink file
        mkdir(os.path.dirname(symlink_path))
        # If the symlink is already there, remove it (multiple copy(*.h) copy(*.dll))
        if os.path.islink(symlink_path):
            os.unlink(symlink_path)
        os.symlink(link_dst, symlink_path)
        copied_files.append(symlink_path)
    return copied_files
