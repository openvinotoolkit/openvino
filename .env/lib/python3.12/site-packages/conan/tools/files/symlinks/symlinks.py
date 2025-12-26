import os


def get_symlinks(base_folder):
    """Return the absolute path to the symlink files in base_folder"""
    for (root, dirnames, filenames) in os.walk(base_folder):
        for el in filenames + dirnames:
            fullpath = os.path.join(root, el)
            if os.path.islink(fullpath):
                yield fullpath


def _path_inside(base, folder):
    base = os.path.abspath(base)
    folder = os.path.abspath(folder)
    return os.path.commonprefix([base, folder]) == base


def absolute_to_relative_symlinks(conanfile, base_folder):
    """
    Convert the symlinks with absolute paths into relative ones if they are pointing to a file or
    directory inside the ``base_folder``. Any absolute symlink pointing outside the ``base_folder``    will be ignored.

    :param conanfile: The current recipe object. Always use ``self``.
    :param base_folder: Folder to be scanned.
    """
    for fullpath in get_symlinks(base_folder):
        link_target = os.readlink(fullpath)
        if not os.path.isabs(link_target):
            continue
        folder_of_symlink = os.path.dirname(fullpath)
        if _path_inside(base_folder, link_target):
            os.unlink(fullpath)
            new_link = os.path.relpath(link_target, folder_of_symlink)
            os.symlink(new_link, fullpath)


def remove_external_symlinks(conanfile, base_folder):
    """
    Remove the symlinks to files that point outside the ``base_folder``, no matter if relative or absolute.

    :param conanfile: The current recipe object. Always use ``self``.
    :param base_folder: Folder to be scanned.
    """
    for fullpath in get_symlinks(base_folder):
        link_target = os.readlink(fullpath)
        if not os.path.isabs(link_target):
            link_target = os.path.join(os.path.dirname(fullpath), link_target)
        if not _path_inside(base_folder, link_target):
            os.unlink(fullpath)


def remove_broken_symlinks(conanfile, base_folder=None):
    """
    Remove the broken symlinks, no matter if relative or absolute.

    :param conanfile: The current recipe object. Always use ``self``.
    :param base_folder: Folder to be scanned.
    """
    for fullpath in get_symlinks(base_folder):
        link_target = os.readlink(fullpath)
        if not os.path.isabs(link_target):
            link_target = os.path.join(os.path.dirname(fullpath), link_target)
        if not os.path.exists(link_target):
            os.unlink(fullpath)
