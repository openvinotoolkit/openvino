import os
from typing import List, Tuple

from conan import ConanFile
from conan.api.output import ConanOutput
from conan.cli.printers.graph import print_graph_basic
from conan.internal.cache.cache import PkgCache
from conan.internal.conan_app import ConanApp
from conan.internal.api.export import cmd_export
from conan.internal.methods import run_package_method
from conan.internal.graph.graph import BINARY_BUILD, RECIPE_INCACHE
from conan.api.model import PkgReference, Remote, RecipeReference
from conan.internal.util.files import mkdir


class ExportAPI:
    """ This API provides methods to export artifacts, both recipes and pre-compiled package
    binaries from user folders to the Conan cache, as Conan recipes and Conan package binaries
    """

    def __init__(self, conan_api, helpers):
        self._conan_api = conan_api
        self._helpers = helpers

    def export(self, path, name: str = None, version: str = None, user: str = None,
               channel: str = None, lockfile=None,
               remotes: List[Remote] = None) -> Tuple[RecipeReference, ConanFile]:
        """ Exports a ``conanfile.py`` recipe, together with its associated files to the Conan cache.
        A "recipe-revision" will be computed and assigned.

        :param path: Path to the conanfile to be exported
        :param name: Optional package name. Typically not necessary as it is defined by the recipe
            attribute or dynamically with the ``set_name()`` method.
            If it is defined in recipe and as an argument, but they don't match, an error will be raised.
        :param version: Optional version. It can be defined in the recipe with the version
            attribute or dynamically with the 'set_version()' method.
            If it is defined in recipe and as an argument, but they don't match, an error will be raised.
        :param user: Optional user. Can be defined by recipe attribute.
            If it is defined in recipe and as an argument, but they don't match, an error will be raised.
        :param channel: Optional channel. Can be defined by recipe attribute.
            If it is defined in recipe and as an argument, but they don't match, an error will be raised.
        :param lockfile: Optional, only relevant if the recipe has 'python-requires' to be locked
        :param remotes: Optional, only relevant to resolve 'python-requires' in remotes
        :return: A tuple of the exported RecipeReference and a ConanFile object
        """
        ConanOutput().title("Exporting recipe to the cache")
        app = ConanApp(self._conan_api)
        hook_manager = self._helpers.hook_manager
        return cmd_export(app.loader, app.cache, hook_manager, self._helpers.global_conf, path,
                          name, version, user, channel, graph_lock=lockfile, remotes=remotes)

    def export_pkg_graph(self, path, ref: RecipeReference, profile_host, profile_build,
                         remotes: List[Remote], lockfile=None, is_build_require=False,
                         skip_binaries=False, output_folder=None):
        """Computes a dependency graph for a given configuration, for an already existing (previously
        exported) recipe in the Conan cache. This method computes the full dependency graph, using
        the profiles, lockfile and remotes information as any other install/graph/create command.
        This is necessary in order to compute the "package_id" of the binary being exported
        into the Conan cache.
        The resulting dependency graph can be passed to ``export_pkg()`` method

        :param path: Path to the conanfile.py in the user folder
        :param ref: full RecipeReference, including recipe-revision
        :param profile_host: Profile for the host context
        :param profile_build: Profile for the build context
        :param lockfile: Optional lockfile
        :param remotes: List of Remotes
        :param is_build_require: In case a package intended to be used as a tool-requires
        :param skip_binaries:
        :param output_folder: The folder containing output files, like potential environment scripts
        :return: A Graph object that can be passed to ``export_pkg()`` method
        """
        assert ref.revision, "ref argument must have recipe-revision defined"
        conan_api = self._conan_api
        deps_graph = conan_api.graph.load_graph_consumer(path,
                                                         ref.name, str(ref.version), ref.user,
                                                         ref.channel,
                                                         profile_host=profile_host,
                                                         profile_build=profile_build,
                                                         lockfile=lockfile, remotes=remotes,
                                                         update=None,
                                                         is_build_require=is_build_require)

        print_graph_basic(deps_graph)
        deps_graph.report_graph_error()
        conan_api.graph.analyze_binaries(deps_graph, build_mode=[ref.name], lockfile=lockfile,
                                         remotes=remotes)
        deps_graph.report_graph_error()

        root_node = deps_graph.root
        root_node.ref = ref  # Make sure the root node revision is well defined

        if not skip_binaries:
            # unless the user explicitly opts-out with --skip-binaries, it is necessary to install
            # binaries, in case there are build_requires necessary like tool-requires=cmake
            # and package() method doing ``cmake.install()``
            # for most cases, deps will be in cache already because of a previous "conan install"
            # but if it is not the case, the binaries from remotes will be downloaded
            conan_api.install.install_binaries(deps_graph=deps_graph, remotes=remotes)
        source_folder = os.path.dirname(path)
        conan_api.install.install_consumer(deps_graph=deps_graph, source_folder=source_folder,
                                           output_folder=output_folder)
        return deps_graph

    def export_pkg(self, graph, output_folder=None) -> None:
        """Executes the ``package()`` method of the exported recipe in order to copy the artifacts
        from user folder to the Conan cache package folder

        :param graph: A Graph object
        :param output_folder: Optional folder where generated files like environment scripts
            of dependencies have been installed
        """
        cache = PkgCache(self._conan_api.cache_folder, self._helpers.global_conf)
        hook_manager = self._helpers.hook_manager

        # The graph has to be loaded with build_mode=[ref.name], so that node is not tried
        # to be downloaded from remotes
        # passing here the create_reference=ref argument is useful so the recipe is in "develop",
        # because the "package()" method is in develop=True already
        pkg_node = graph.root
        ref = pkg_node.ref
        source_folder = os.path.dirname(pkg_node.path)
        out = ConanOutput(scope=pkg_node.conanfile.display_name)
        out.info("Exporting binary from user folder to Conan cache")
        conanfile = pkg_node.conanfile

        package_id = pkg_node.package_id
        assert package_id is not None
        out.info("Packaging to %s" % package_id)
        pref = PkgReference(ref, package_id)
        pkg_layout = cache.create_build_pkg_layout(pref)

        conanfile.folders.set_base_folders(source_folder, output_folder)
        dest_package_folder = pkg_layout.package()
        conanfile.folders.set_base_package(dest_package_folder)
        mkdir(pkg_layout.metadata())
        conanfile.folders.set_base_pkg_metadata(pkg_layout.metadata())

        with pkg_layout.set_dirty_context_manager():
            prev = run_package_method(conanfile, package_id, hook_manager, ref)

        pref = PkgReference(pref.ref, pref.package_id, prev)
        pkg_layout.reference = pref
        cache.assign_prev(pkg_layout)
        pkg_node.prev = prev
        pkg_node.pref_timestamp = pref.timestamp  # assigned by assign_prev
        pkg_node.recipe = RECIPE_INCACHE
        pkg_node.binary = BINARY_BUILD
        # Make sure folder is updated
        final_folder = pkg_layout.package()
        conanfile.folders.set_base_package(final_folder)
        out.info(f"Package folder {final_folder}")
        out.success("Exported package binary")
