import logging
import os
import shutil

import patch_ng
import yaml

from conan.errors import ConanException
from conan.internal.paths import DATA_YML
from conan.internal.util.files import mkdir, load, save


class PatchLogHandler(logging.Handler):
    def __init__(self, scoped_output, patch_file):
        logging.Handler.__init__(self, logging.DEBUG)
        self._scoped_output = scoped_output
        self.patchname = patch_file or "patch_ng"

    def emit(self, record):
        logstr = self.format(record)
        if record.levelno == logging.WARN:
            self._scoped_output.warning("%s: %s" % (self.patchname, logstr))
        else:
            self._scoped_output.info("%s: %s" % (self.patchname, logstr))


def patch(conanfile, base_path=None, patch_file=None, patch_string=None, strip=0, fuzz=False, **kwargs):
    """
    Applies a diff from file (patch_file) or string (patch_string) in the conanfile.source_folder
    directory. The folder containing the sources can be customized with the self.folders attribute
    in the layout(self) method.

    :param conanfile: the current recipe, always pass 'self'
    :param base_path: The path is a relative path to conanfile.export_sources_folder unless an
           absolute path is provided.
    :param patch_file: Patch file that should be applied. The path is relative to the
           conanfile.source_folder unless an absolute path is provided.
    :param patch_string: Patch string that should be applied.
    :param strip: Number of folders to be stripped from the path.
    :param fuzz: Should accept fuzzy patches.
    :param kwargs: Extra parameters that can be added and will contribute to output information
    """

    patch_type = kwargs.get('patch_type') or ("file" if patch_file else "string")
    patch_description = kwargs.get('patch_description')

    if patch_type or patch_description:
        patch_type_str = ' ({})'.format(patch_type) if patch_type else ''
        patch_description_str = ': {}'.format(patch_description) if patch_description else ''
        conanfile.output.info('Apply patch{}{}'.format(patch_type_str, patch_description_str))

    patchlog = logging.getLogger("patch_ng")
    patchlog.handlers = []
    patchlog.addHandler(PatchLogHandler(conanfile.output, patch_file))

    if patch_file:
        # trick *1: patch_file path could be absolute (e.g. conanfile.build_folder), in that case
        # the join does nothing and works.
        patch_path = os.path.join(conanfile.export_sources_folder, patch_file)
        patchset = patch_ng.fromfile(patch_path)
    else:
        patchset = patch_ng.fromstring(patch_string.encode())

    if not patchset:
        raise ConanException("Failed to parse patch: %s" % (patch_file if patch_file else "string"))

    # trick *1
    root = os.path.join(conanfile.source_folder, base_path) if base_path else conanfile.source_folder
    if not patchset.apply(strip=strip, root=root, fuzz=fuzz):
        raise ConanException("Failed to apply patch: %s" % patch_file)


def apply_conandata_patches(conanfile):
    """
    Applies patches stored in ``conanfile.conan_data`` (read from ``conandata.yml`` file).
    It will apply all the patches under ``patches`` entry that matches the given
    ``conanfile.version``. If versions are not defined in ``conandata.yml`` it will apply all the
    patches directly under ``patches`` keyword.

    The key entries will be passed as kwargs to the ``patch`` function.
    """
    if conanfile.conan_data is None:
        raise ConanException("conandata.yml not defined")

    patches = conanfile.conan_data.get('patches')
    if patches is None:
        conanfile.output.info("apply_conandata_patches(): No patches defined in conandata")
        return

    if isinstance(patches, dict):
        assert conanfile.version, "Can only be applied if conanfile.version is already defined"
        entries = patches.get(str(conanfile.version), [])
        if entries is None:
            conanfile.output.warning(f"apply_conandata_patches(): No patches defined for version {conanfile.version} in conandata.yml")
            return
    elif isinstance(patches, list):
        entries = patches
    else:
        raise ConanException("conandata.yml 'patches' should be a list or a dict {version: list}")
    for it in entries:
        if "patch_file" in it:
            # The patch files are located in the root src
            entry = it.copy()
            patch_file = entry.pop("patch_file")
            patch_file_path = os.path.join(conanfile.export_sources_folder, patch_file)
            if "patch_description" not in entry:
                entry["patch_description"] = patch_file
            patch(conanfile, patch_file=patch_file_path, **entry)
        elif "patch_string" in it:
            patch(conanfile, **it)
        elif "patch_user" in it:
            pass  # This will be managed directly by users, can be a command or a script execution
        else:
            raise ConanException("The 'conandata.yml' file needs a 'patch_file' or 'patch_string'"
                                 " entry for every patch to be applied")


def export_conandata_patches(conanfile):
    """
    Exports patches stored in 'conanfile.conan_data' (read from 'conandata.yml' file). It will export
    all the patches under 'patches' entry that matches the given 'conanfile.version'. If versions are
    not defined in 'conandata.yml' it will export all the patches directly under 'patches' keyword.
    """
    if conanfile.conan_data is None:
        raise ConanException("conandata.yml not defined")

    conanfile_patches = conanfile.conan_data.get('patches')

    def _handle_patches(patches, patches_folder):
        if patches is None:
            conanfile.output.info("export_conandata_patches(): No patches defined in conandata")
            return

        if isinstance(patches, dict):
            assert conanfile.version, "Can only be exported if conanfile.version is already defined"
            entries = patches.get(conanfile.version, [])
            if entries is None:
                conanfile.output.warning("export_conandata_patches(): No patches defined for "
                                         f"version {conanfile.version} in conandata.yml")
                return
        elif isinstance(patches, list):
            entries = patches
        else:
            raise ConanException("conandata.yml 'patches' should be a list or a dict "
                                 "{version: list}")
        for it in entries:
            patch_file = it.get("patch_file")
            if patch_file:
                src = os.path.join(patches_folder, patch_file)
                dst = os.path.join(conanfile.export_sources_folder, patch_file)
                if not os.path.exists(src):
                    raise ConanException(f"Patch file does not exist: '{src}'")
                mkdir(os.path.dirname(dst))
                shutil.copy2(src, dst)
        return entries

    _handle_patches(conanfile_patches, conanfile.recipe_folder)

    extra_path = conanfile.conf.get("core.sources.patch:extra_path")
    if extra_path:
        if not os.path.isdir(extra_path):
            raise ConanException(f"Patches extra path '{extra_path}' does not exist")
        pkg_path = os.path.join(extra_path, conanfile.name)
        if not os.path.isdir(pkg_path):
            return
        data_path = os.path.join(pkg_path, DATA_YML)
        try:
            data = yaml.safe_load(load(data_path))
        except Exception as e:
            raise ConanException("Invalid yml format at {}: {}".format(data_path, e))
        data = data or {}
        conanfile.output.info(f"Applying extra patches 'core.sources.patch:extra_path': {data_path}")
        new_patches = _handle_patches(data.get('patches'), pkg_path)

        # Update the CONANDATA.YML
        conanfile_patches = conanfile_patches or {}
        conanfile_patches.setdefault(conanfile.version, []).extend(new_patches)

        conanfile.conan_data['patches'] = conanfile_patches
        # Saving in the EXPORT folder
        conanfile_data_path = os.path.join(conanfile.export_folder, DATA_YML)
        new_conandata_yml = yaml.safe_dump(conanfile.conan_data, default_flow_style=False)
        save(conanfile_data_path, new_conandata_yml)
