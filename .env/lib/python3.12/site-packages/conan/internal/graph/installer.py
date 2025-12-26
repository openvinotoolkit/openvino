import os
import shutil
from multiprocessing.pool import ThreadPool

from conan.api.output import ConanOutput, Color
from conan.internal.methods import run_build_method, run_package_method
from conan.internal.api.install.generators import write_generators
from conan.internal.graph.graph import (BINARY_BUILD, BINARY_CACHE, BINARY_DOWNLOAD,
                                        BINARY_EDITABLE, BINARY_UPDATE, BINARY_EDITABLE_BUILD,
                                        BINARY_SKIP)
from conan.internal.graph.install_graph import InstallGraph
from conan.internal.source import retrieve_exports_sources, config_source
from conan.internal.errors import conanfile_remove_attr, conanfile_exception_formatter
from conan.errors import ConanException
from conan.internal.model.cpp_info import CppInfo, MockInfoProperty
from conan.api.model import PkgReference
from conan.internal.paths import CONANINFO
from conan.internal.util import cpu_count
from conan.internal.util.files import clean_dirty, is_dirty, mkdir, rmdir, save, set_dirty, chdir


def build_id(conan_file):
    if hasattr(conan_file, "build_id"):
        # construct new ConanInfo
        build_id_info = conan_file.info.clone()
        conan_file.info_build = build_id_info
        # effectively call the user function to change the package values
        with conanfile_exception_formatter(conan_file, "build_id"):
            conan_file.build_id()
        # compute modified ID
        return build_id_info.package_id()
    return None


class _PackageBuilder:

    def __init__(self, app, hook_manager):
        self._cache = app.cache
        self._hook_manager = hook_manager
        self._remote_manager = app.remote_manager
        self._home_folder = app.cache_folder

    def _get_build_folder(self, conanfile, package_layout):
        # Build folder can use a different package_ID if build_id() is defined.
        # This function decides if the build folder should be re-used (not build again)
        # and returns the build folder
        skip_build = False
        build_folder = package_layout.build()
        recipe_build_id = build_id(conanfile)
        pref = package_layout.reference
        if recipe_build_id is not None and pref.package_id != recipe_build_id:
            conanfile.output.info(f"build_id() computed {recipe_build_id}")
            # check if we already have a package with the calculated build_id
            recipe_ref = pref.ref
            build_prev = self._cache.get_matching_build_id(recipe_ref, recipe_build_id)
            if build_prev is None:  # Only store build_id of the first one actually building it
                package_layout.build_id = recipe_build_id
            build_prev = build_prev or pref

            # We are trying to build a package id different from the one that has the
            # build_folder but belongs to the same recipe revision, so reuse the build_folder
            # from the one that is already build
            if build_prev.package_id != pref.package_id:
                other_pkg_layout = self._cache.pkg_layout(build_prev)
                build_folder = other_pkg_layout.build()
                skip_build = True

        if is_dirty(build_folder):
            conanfile.output.warning("Build folder is dirty, removing it: %s" % build_folder)
            rmdir(build_folder)
            clean_dirty(build_folder)
            skip_build = False

        if skip_build and os.path.exists(build_folder):
            conanfile.output.info("Won't be built, using previous build folder as defined "
                                  f"in build_id(): {build_folder}")

        return build_folder, skip_build

    @staticmethod
    def _copy_sources(conanfile, source_folder, build_folder):
        # Copies the sources to the build-folder, unless no_copy_source is defined
        rmdir(build_folder)
        if not getattr(conanfile, 'no_copy_source', False):
            conanfile.output.info('Copying sources to build folder')
            try:
                shutil.copytree(source_folder, build_folder, symlinks=True)
            except Exception as e:
                raise ConanException(f"{e}\nError copying sources to build folder")

    def _build(self, conanfile, pref):
        write_generators(conanfile, self._hook_manager, self._home_folder)

        try:
            run_build_method(conanfile, self._hook_manager)
            conanfile.output.success("Package '%s' built" % pref.package_id)
            conanfile.output.info("Build folder %s" % conanfile.build_folder)
        except Exception as exc:
            conanfile.output.error(f"\nPackage '{pref.package_id}' build failed",
                                   error_type="exception")
            conanfile.output.warning("Build folder %s" % conanfile.build_folder)
            if isinstance(exc, ConanException):
                raise exc
            raise ConanException(exc)

    def _package(self, conanfile, pref):
        # Creating ***info.txt files
        save(os.path.join(conanfile.folders.base_build, CONANINFO), conanfile.info.dumps())

        package_id = pref.package_id
        # Do the actual copy, call the conanfile.package() method
        # While installing, the infos goes to build folder
        prev = run_package_method(conanfile, package_id, self._hook_manager, pref.ref)

        # FIXME: Conan 2.0 Clear the registry entry (package ref)
        return prev

    def build_package(self, node, package_layout):
        conanfile = node.conanfile
        pref = node.pref

        # TODO: cache2.0 fix this
        recipe_layout = self._cache.recipe_layout(pref.ref)

        base_source = recipe_layout.source()
        base_package = package_layout.package()

        base_build, skip_build = self._get_build_folder(conanfile, package_layout)

        # PREPARE SOURCES
        if not skip_build:
            # TODO: cache2.0 check locks
            # with package_layout.conanfile_write_lock(self._output):
            set_dirty(base_build)
            self._copy_sources(conanfile, base_source, base_build)
            mkdir(base_build)

        # BUILD & PACKAGE
        # TODO: cache2.0 check locks
        # with package_layout.conanfile_read_lock(self._output):
        with chdir(base_build):
            try:
                src = base_source if getattr(conanfile, 'no_copy_source', False) else base_build
                conanfile.folders.set_base_source(src)
                conanfile.folders.set_base_build(base_build)
                conanfile.folders.set_base_package(base_package)
                # In local cache, generators folder always in build_folder
                conanfile.folders.set_base_generators(base_build)
                conanfile.folders.set_base_pkg_metadata(package_layout.metadata())

                if not skip_build:
                    conanfile.output.info('Building your package in %s' % base_build)
                    # In local cache, install folder always is build_folder
                    self._build(conanfile, pref)
                    clean_dirty(base_build)

                prev = self._package(conanfile, pref)
                assert prev
                node.prev = prev
            except ConanException as exc:  # TODO: Remove this? unnecessary?
                raise exc

        return node.pref


class BinaryInstaller:
    """ main responsible of retrieving binary packages or building them from source
    locally in case they are not found in remotes
    """

    def __init__(self, app, global_conf, editable_packages, hook_manager):
        self._app = app
        self._editable_packages = editable_packages
        self._cache = app.cache
        self._remote_manager = app.remote_manager
        self._hook_manager = hook_manager
        self._global_conf = global_conf
        self._home_folder = app.cache_folder

    def _install_source(self, node, remotes, need_conf=False):
        conanfile = node.conanfile
        download_source = conanfile.conf.get("tools.build:download_source", check_type=bool)

        if not download_source and (need_conf or node.binary != BINARY_BUILD):
            return

        conanfile = node.conanfile
        if node.binary == BINARY_EDITABLE:
            return

        recipe_layout = self._cache.recipe_layout(node.ref)
        export_source_folder = recipe_layout.export_sources()
        source_folder = recipe_layout.source()

        retrieve_exports_sources(self._remote_manager, recipe_layout, conanfile, node.ref, remotes)

        conanfile.folders.set_base_source(source_folder)
        conanfile.folders.set_base_export_sources(source_folder)
        conanfile.folders.set_base_recipe_metadata(recipe_layout.metadata())
        config_source(export_source_folder, conanfile, self._hook_manager)

    @staticmethod
    def install_system_requires(graph, only_info=False, install_order=None):
        if install_order is None:
            install_graph = InstallGraph(graph)
            install_order = install_graph.install_order()

        for level in install_order:
            for install_reference in level:
                for package in install_reference.packages.values():
                    if not only_info and package.binary == BINARY_SKIP:
                        continue
                    conanfile = package.nodes[0].conanfile
                    # TODO: Refactor magic strings and use _SystemPackageManagerTool.mode_xxx ones
                    mode = conanfile.conf.get("tools.system.package_manager:mode")
                    if only_info and mode is None:
                        continue
                    if hasattr(conanfile, "system_requirements"):
                        with conanfile_exception_formatter(conanfile, "system_requirements"):
                            conanfile.system_requirements()
                    for n in package.nodes:
                        n.conanfile.system_requires = conanfile.system_requires

        conanfile = graph.root.conanfile
        mode = conanfile.conf.get("tools.system.package_manager:mode")
        if only_info and mode is None:
            return
        if hasattr(conanfile, "system_requirements"):
            with conanfile_exception_formatter(conanfile, "system_requirements"):
                conanfile.system_requirements()

    def install_sources(self, graph, remotes):
        install_graph = InstallGraph(graph)
        install_order = install_graph.install_order()

        for level in install_order:
            for install_reference in level:
                for package in install_reference.packages.values():
                    self._install_source(package.nodes[0], remotes, need_conf=True)

    def install(self, deps_graph, remotes, install_order=None):
        assert not deps_graph.error, "This graph cannot be installed: {}".format(deps_graph)
        if install_order is None:
            install_graph = InstallGraph(deps_graph)
            install_graph.raise_errors()
            install_order = install_graph.install_order()

        ConanOutput().title("Installing packages")

        package_count = sum([sum(len(install_reference.packages.values())
                                 for level in install_order
                                 for install_reference in level)])
        handled_count = 1

        self._download_bulk(install_order)
        for level in install_order:
            for install_reference in level:
                for package in install_reference.packages.values():
                    self._install_source(package.nodes[0], remotes)
                    self._handle_package(package, install_reference, handled_count, package_count)
                    handled_count += 1

        MockInfoProperty.message()

    def _download_bulk(self, install_order):
        """ executes the download of packages (both download and update), only once for a given
        PREF
        """
        downloads = []
        for level in install_order:
            for node in level:
                for package in node.packages.values():
                    if package.binary in (BINARY_UPDATE, BINARY_DOWNLOAD):
                        downloads.append(package)
        if not downloads:
            return

        download_count = len(downloads)
        plural = 's' if download_count != 1 else ''
        ConanOutput().subtitle(f"Downloading {download_count} package{plural}")
        parallel = self._global_conf.get("core.download:parallel", check_type=int,
                                         default=cpu_count())
        if parallel:  # User can define core.download:parallel=0 to deactivate parallelism
            ConanOutput().info("Downloading binary packages in %s parallel threads" % parallel)
            thread_pool = ThreadPool(parallel)
            thread_pool.map(self._download_pkg, downloads)
            thread_pool.close()
            thread_pool.join()
        else:
            for node in downloads:
                self._download_pkg(node)

    def _download_pkg(self, package):
        node = package.nodes[0]
        assert node.pref.revision is not None
        assert node.pref.timestamp is not None
        self._remote_manager.get_package(node.pref, node.binary_remote)

    def _handle_package(self, package, install_reference, handled_count, total_count):
        if package.binary in (BINARY_EDITABLE, BINARY_EDITABLE_BUILD):
            self._handle_node_editable(package)
            return

        assert package.binary in (BINARY_CACHE, BINARY_BUILD, BINARY_DOWNLOAD, BINARY_UPDATE)
        assert install_reference.ref.revision is not None, "Installer should receive RREV always"

        pref = PkgReference(install_reference.ref, package.package_id, package.prev)

        if package.binary == BINARY_BUILD:
            assert pref.revision is None
            ConanOutput()\
                .subtitle(f"Installing package {pref.ref} ({handled_count} of {total_count})")
            ConanOutput(scope=str(pref.ref))\
                .highlight("Building from source")\
                .info(f"Package {pref}")
            compact_dumps = package.nodes[0].conanfile.info.summarize_compact()
            for line in compact_dumps:
                ConanOutput(scope=str(pref.ref)).info(line, fg=Color.BRIGHT_GREEN)
            package_layout = self._cache.create_build_pkg_layout(pref)
            self._handle_node_build(package, package_layout)
            # Just in case it was recomputed
            package.package_id = package.nodes[0].pref.package_id  # Just in case it was recomputed
            package.prev = package.nodes[0].pref.revision
            package.binary = package.nodes[0].binary
            pref = PkgReference(install_reference.ref, package.package_id, package.prev)
        else:
            assert pref.revision is not None
            package_layout = self._cache.pkg_layout(pref)
            if package.binary == BINARY_CACHE:
                node = package.nodes[0]
                pref = node.pref
                self._cache.update_package_lru(pref)
                assert node.prev, "PREV for %s is None" % str(pref)
                msg = f'Already installed! ({handled_count} of {total_count})'
                node.conanfile.output.success(msg)

        # Make sure that all nodes with same pref compute package_info()
        pkg_folder = package_layout.package()
        pkg_metadata = package_layout.metadata()
        assert os.path.isdir(pkg_folder), "Pkg '%s' folder must exist: %s" % (str(pref), pkg_folder)
        for n in package.nodes:
            n.prev = pref.revision  # Make sure the prev is assigned
            conanfile = n.conanfile
            # Call the info method
            conanfile.folders.set_base_package(pkg_folder)
            conanfile.folders.set_base_pkg_metadata(pkg_metadata)
            self._call_finalize_method(conanfile, package_layout.finalize())
            # Use package_folder which has been updated previously by install_method if necessary
            self._call_package_info(conanfile, conanfile.package_folder, is_editable=False)

    def _handle_node_editable(self, install_node):
        # It will only run generation
        node = install_node.nodes[0]
        conanfile = node.conanfile
        ref = node.ref
        editable = self._editable_packages.get(ref)
        conanfile_path = editable["path"]
        output_folder = editable.get("output_folder")

        base_path = os.path.dirname(conanfile_path)

        conanfile.folders.set_base_folders(base_path, output_folder)

        if node.binary == BINARY_EDITABLE_BUILD:
            output = conanfile.output
            output.info("Rewriting files of editable package "
                        "'{}' at '{}'".format(conanfile.name, conanfile.generators_folder))
            write_generators(conanfile, self._hook_manager, self._home_folder)
            run_build_method(conanfile, self._hook_manager)

        rooted_base_path = base_path if conanfile.folders.root is None else \
            os.path.normpath(os.path.join(base_path, conanfile.folders.root))

        for node in install_node.nodes:
            # Get source of information
            conanfile = node.conanfile
            # New editables mechanism based on Folders
            conanfile.folders.set_base_package(output_folder or rooted_base_path)
            conanfile.folders.set_base_folders(base_path, output_folder)
            conanfile.folders.set_base_pkg_metadata(os.path.join(conanfile.build_folder, "metadata"))
            # Need a temporary package revision for package_revision_mode
            # Cannot be PREV_UNKNOWN otherwise the consumers can't compute their packageID
            node.prev = "editable"
            # TODO: Check this base_path usage for editable when not defined
            self._call_package_info(conanfile, package_folder=rooted_base_path, is_editable=True)

    def _handle_node_build(self, package, pkg_layout):
        node = package.nodes[0]
        pref = node.pref
        assert pref.package_id, "Package-ID without value"
        assert pkg_layout, "The pkg_layout should be declared here"
        assert node.binary == BINARY_BUILD
        assert node.prev is None, "PREV for %s to be built should be None" % str(pref)

        with pkg_layout.package_lock():
            pkg_layout.package_remove()
            with pkg_layout.set_dirty_context_manager():
                builder = _PackageBuilder(self._app, self._hook_manager)
                pref = builder.build_package(node, pkg_layout)
            assert node.prev, "Node PREV shouldn't be empty"
            assert node.pref.revision, "Node PREF revision shouldn't be empty"
            assert pref.revision is not None, "PREV for %s to be built is None" % str(pref)
            # at this point the package reference should be complete
            pkg_layout.reference = pref
            self._cache.assign_prev(pkg_layout)
            # Make sure the current conanfile.folders is updated (it is later in package_info(),
            # but better make sure here, and be able to report the actual folder in case
            # something fails)
            node.conanfile.folders.set_base_package(pkg_layout.package())
            node.conanfile.output.success("Package folder %s" % node.conanfile.package_folder)

    def _call_package_info(self, conanfile, package_folder, is_editable):

        with chdir(package_folder):
            with conanfile_exception_formatter(conanfile, "package_info"):
                self._hook_manager.execute("pre_package_info", conanfile=conanfile)

                if hasattr(conanfile, "package_info"):
                    with conanfile_remove_attr(conanfile, ['info', "source_folder", "build_folder"],
                                               "package_info"):
                        MockInfoProperty.package = str(conanfile)
                        conanfile.package_info()

                # TODO: Check this package_folder usage for editable when not defined
                conanfile.cpp.package.set_relative_base_folder(package_folder)

                if is_editable:
                    # Adjust the folders of the layout to consolidate the rootfolder of the
                    # cppinfos inside

                    # convert directory entries to be relative to the declared folders.build
                    build_cppinfo = conanfile.cpp.build
                    build_cppinfo.set_relative_base_folder(conanfile.build_folder)
                    conanfile.layouts.build.set_relative_base_folder(conanfile.build_folder)

                    # convert directory entries to be relative to the declared folders.source
                    source_cppinfo = conanfile.cpp.source
                    source_cppinfo.set_relative_base_folder(conanfile.source_folder)
                    conanfile.layouts.source.set_relative_base_folder(conanfile.source_folder)

                    full_editable_cppinfo = CppInfo()
                    full_editable_cppinfo.merge(source_cppinfo)
                    full_editable_cppinfo.merge(build_cppinfo)
                    # In editables if we defined anything in the cpp infos we want to discard
                    # the one defined in the conanfile cpp_info
                    conanfile.cpp_info.merge(full_editable_cppinfo, overwrite=True)

                    # Paste the editable cpp_info but prioritizing it, only if a
                    # variable is not declared at build/source, the package will keep the value
                    full_buildenv_info = conanfile.layouts.source.buildenv_info.copy()
                    full_buildenv_info.compose_env(conanfile.layouts.build.buildenv_info)
                    full_buildenv_info.compose_env(conanfile.buildenv_info)
                    conanfile.buildenv_info = full_buildenv_info

                    full_runenv_info = conanfile.layouts.source.runenv_info.copy()
                    full_runenv_info.compose_env(conanfile.layouts.build.runenv_info)
                    full_runenv_info.compose_env(conanfile.runenv_info)
                    conanfile.runenv_info = full_runenv_info

                    full_conf_info = conanfile.layouts.source.conf_info.copy()
                    full_conf_info.compose_conf(conanfile.layouts.build.conf_info)
                    full_conf_info.compose_conf(conanfile.conf_info)
                    conanfile.conf_info = full_conf_info
                else:
                    conanfile.layouts.package.set_relative_base_folder(conanfile.package_folder)
                    conanfile.buildenv_info.compose_env(conanfile.layouts.package.buildenv_info)
                    conanfile.runenv_info.compose_env(conanfile.layouts.package.runenv_info)
                    conanfile.conf_info.compose_conf(conanfile.layouts.package.conf_info)

                self._hook_manager.execute("post_package_info", conanfile=conanfile)

        conanfile.cpp_info.check_component_requires(conanfile)
        try:
            conanfile.conf_info.validate()
        except ConanException as e:
            raise ConanException(f"{conanfile}: Error in package_info() method:\n\t{str(e)}")

    @staticmethod
    def _call_finalize_method(conanfile, finalize_folder):
        if hasattr(conanfile, "finalize"):
            conanfile.folders.set_finalize_folder(finalize_folder)
            if not os.path.exists(finalize_folder):
                mkdir(finalize_folder)
                conanfile.output.highlight("Calling finalize()")
                with conanfile_exception_formatter(conanfile, "finalize"):
                    with conanfile_remove_attr(conanfile, ['cpp_info', 'settings', 'options'],
                                               'finalize'):
                        conanfile.finalize()

            conanfile.output.success(f"Finalized folder {finalize_folder}")
