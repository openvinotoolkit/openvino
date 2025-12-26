import os

from conan.api.output import ConanOutput
from conan.internal.methods import run_source_method
from conan.tools.env import VirtualBuildEnv
from conan.internal.errors import NotFoundException
from conan.errors import ConanException
from conan.internal.util.files import is_dirty, mkdir, rmdir, set_dirty_context_manager, merge_directories, clean_dirty


def _try_get_sources(ref, remote_manager, recipe_layout, remote):
    try:
        remote_manager.get_recipe_sources(ref, recipe_layout, remote)
    except NotFoundException:
        return
    except Exception as e:
        msg = ("The '%s' package has 'exports_sources' but sources not found in local cache.\n"
               "Probably it was installed from a remote that is no longer available.\n"
               % str(ref))
        raise ConanException("\n".join([str(e), msg]))
    return remote


def retrieve_exports_sources(remote_manager, recipe_layout, conanfile, ref, remotes):
    """ the "exports_sources" sources are not retrieved unless necessary to build. In some
    occassions, conan needs to get them too, like if uploading to a server, to keep the recipes
    complete
    """
    if conanfile.exports_sources is None and not hasattr(conanfile, "export_sources"):
        return None

    export_sources_folder = recipe_layout.export_sources()
    if os.path.exists(export_sources_folder):
        return None

    for r in remotes:
        sources_remote = _try_get_sources(ref, remote_manager, recipe_layout, r)
        if sources_remote:
            break
    else:
        msg = ("The '%s' package has 'exports_sources' but sources not found in local cache.\n"
               "Probably it was installed from a remote that is no longer available.\n"
               % str(ref))
        raise ConanException(msg)

    ConanOutput(scope=str(ref)).info("Sources downloaded from '{}'".format(sources_remote.name))


def config_source(export_source_folder, conanfile, hook_manager):
    """ Implements the sources configuration when a package is going to be built in the
    local cache:
    - remove old sources if dirty
    - do a copy of the exports_sources folders to the source folder in the cache
    - run the source() recipe method
    """

    if is_dirty(conanfile.folders.base_source):
        conanfile.output.warning("Trying to remove corrupted source folder")
        conanfile.output.warning("This can take a while for big packages")
        rmdir(conanfile.folders.base_source)
        clean_dirty(conanfile.folders.base_source)

    if not os.path.exists(conanfile.folders.base_source):  # No source folder, need to get it
        with set_dirty_context_manager(conanfile.folders.base_source):
            mkdir(conanfile.source_folder)
            mkdir(conanfile.recipe_metadata_folder)

            # First of all get the exported scm sources (if auto) or clone (if fixed)
            # Now move the export-sources to the right location
            merge_directories(export_source_folder, conanfile.folders.base_source)
            if getattr(conanfile, "source_buildenv", False):
                with VirtualBuildEnv(conanfile, auto_generate=True).vars().apply():
                    run_source_method(conanfile, hook_manager)
            else:
                run_source_method(conanfile, hook_manager)
