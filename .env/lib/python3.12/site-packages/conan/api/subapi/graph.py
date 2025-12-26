from conan.api.output import ConanOutput
from conan.internal.conan_app import ConanApp, ConanBasicApp
from conan.internal.model.recipe_ref import ref_matches
from conan.internal.graph.graph import Node, RECIPE_CONSUMER, CONTEXT_HOST, RECIPE_VIRTUAL, \
    CONTEXT_BUILD, BINARY_MISSING, DepsGraph
from conan.internal.graph.graph_binaries import GraphBinariesAnalyzer
from conan.internal.graph.graph_builder import DepsGraphBuilder
from conan.internal.graph.install_graph import InstallGraph, ProfileArgs
from conan.internal.graph.profile_node_definer import initialize_conanfile_profile, consumer_definer
from conan.errors import ConanException
from conan.api.model import RecipeReference


class GraphAPI:

    def __init__(self, conan_api, helpers):
        self._conan_api = conan_api
        self._helpers = helpers

    def _load_root_consumer_conanfile(self, path, profile_host, profile_build,
                                      name=None, version=None, user=None, channel=None,
                                      update=None, remotes=None, lockfile=None,
                                      is_build_require=False):
        app = ConanApp(self._conan_api)

        if path.endswith(".py"):
            conanfile = app.loader.load_consumer(path,
                                                 name=name,
                                                 version=version,
                                                 user=user,
                                                 channel=channel,
                                                 graph_lock=lockfile,
                                                 remotes=remotes,
                                                 update=update)
            ref = RecipeReference(conanfile.name, conanfile.version,
                                  conanfile.user, conanfile.channel)
            context = CONTEXT_BUILD if is_build_require else CONTEXT_HOST
            # Here, it is always the "host" context because it is the base, not the current node one
            initialize_conanfile_profile(conanfile, profile_build, profile_host, CONTEXT_HOST,
                                         is_build_require, ref)
            if ref.name:
                profile_host.options.scope(ref)
            root_node = Node(ref, conanfile, context=context, recipe=RECIPE_CONSUMER, path=path)
            root_node.should_build = True  # It is a consumer, this is something we are building
        else:
            conanfile = app.loader.load_conanfile_txt(path)
            consumer_definer(conanfile, profile_host, profile_build)
            root_node = Node(None, conanfile, context=CONTEXT_HOST, recipe=RECIPE_CONSUMER,
                             path=path)
        return root_node

    def load_root_test_conanfile(self, path, tested_reference, profile_host, profile_build,
                                 update=None, remotes=None, lockfile=None,
                                 tested_python_requires=None):
        """ Create and initialize a root node from a test_package/conanfile.py consumer

        :param tested_python_requires: the reference of the ``python_require`` to be tested
        :param lockfile: Might be good to lock python-requires, build-requires
        :param path: The full path to the test_package/conanfile.py being used
        :param tested_reference: The full RecipeReference of the tested package
        :param profile_host:
        :param profile_build:
        :param update:
        :param remotes:
        :return: a graph Node, recipe=RECIPE_CONSUMER
        """

        app = ConanApp(self._conan_api)
        # necessary for correct resolution and update of remote python_requires

        loader = app.loader
        profile_host.options.scope(tested_reference)

        # do not try apply lock_python_requires for test_package/conanfile.py consumer
        conanfile = loader.load_consumer(path, user=tested_reference.user,
                                         channel=tested_reference.channel,
                                         graph_lock=lockfile, remotes=remotes,
                                         tested_python_requires=tested_python_requires,
                                         update=update)
        initialize_conanfile_profile(conanfile, profile_build, profile_host, CONTEXT_HOST, False)
        conanfile.display_name = "%s (test package)" % str(tested_reference)
        conanfile.output.scope = conanfile.display_name
        conanfile.tested_reference_str = repr(tested_reference)

        ref = RecipeReference(conanfile.name, conanfile.version, tested_reference.user,
                              tested_reference.channel)
        root_node = Node(ref, conanfile, recipe=RECIPE_CONSUMER, context=CONTEXT_HOST, path=path)
        return root_node

    def _load_root_virtual_conanfile(self, profile_host, profile_build, requires, tool_requires,
                                     lockfile, remotes, update, check_updates=False,
                                     python_requires=None):
        if not python_requires and not requires and not tool_requires:
            raise ConanException("Provide requires or tool_requires")
        app = ConanApp(self._conan_api)
        conanfile = app.loader.load_virtual(requires=requires,
                                            tool_requires=tool_requires,
                                            python_requires=python_requires,
                                            graph_lock=lockfile, remotes=remotes,
                                            update=update, check_updates=check_updates)

        consumer_definer(conanfile, profile_host, profile_build)
        root_node = Node(ref=None, conanfile=conanfile, context=CONTEXT_HOST, recipe=RECIPE_VIRTUAL)
        return root_node

    @staticmethod
    def _scope_options(profile, requires, tool_requires):
        """
        Command line helper to scope options when ``command -o myoption=myvalue`` is used,
        that needs to be converted to "-o pkg:myoption=myvalue". The "pkg" value will be
        computed from the given requires/tool_requires

        This is legacy, as options should always be scoped now
        """
        if requires and len(requires) == 1 and not tool_requires:
            ref = requires[0]
            if str(ref.version).startswith("["):
                ref = ref.copy()
                ref.version = "*"
            profile.options.scope(ref)

    def load_graph_requires(self, requires, tool_requires, profile_host, profile_build,
                            lockfile, remotes, update, check_updates=False, python_requires=None):
        requires = [RecipeReference.loads(r) if isinstance(r, str) else r for r in requires] \
            if requires else None
        tool_requires = [RecipeReference.loads(r) if isinstance(r, str) else r
                         for r in tool_requires] if tool_requires else None

        self._scope_options(profile_host, requires=requires, tool_requires=tool_requires)
        root_node = self._load_root_virtual_conanfile(requires=requires, tool_requires=tool_requires,
                                                      profile_host=profile_host,
                                                      profile_build=profile_build,
                                                      lockfile=lockfile, remotes=remotes,
                                                      update=update,
                                                      python_requires=python_requires)

        if not requires and not tool_requires and python_requires is not None:
            # This only happens at `conan create` for python-requires, the graph is not needed
            # in fact, it can cause errors, if tool-requires injected
            dep_graph = DepsGraph()
            dep_graph.add_node(root_node)
            return dep_graph
        # check_updates = args.check_updates if "check_updates" in args else False
        deps_graph = self.load_graph(root_node, profile_host=profile_host,
                                     profile_build=profile_build,
                                     lockfile=lockfile,
                                     remotes=remotes,
                                     update=update,
                                     check_update=check_updates)
        return deps_graph

    def load_graph_consumer(self, path, name, version, user, channel,
                            profile_host, profile_build, lockfile, remotes, update,
                            check_updates=False, is_build_require=False):
        root_node = self._load_root_consumer_conanfile(path, profile_host, profile_build,
                                                       name=name, version=version, user=user,
                                                       channel=channel, lockfile=lockfile,
                                                       remotes=remotes, update=update,
                                                       is_build_require=is_build_require)

        deps_graph = self.load_graph(root_node, profile_host=profile_host,
                                     profile_build=profile_build, lockfile=lockfile,
                                     remotes=remotes, update=update, check_update=check_updates)
        return deps_graph

    def load_graph(self, root_node, profile_host, profile_build, lockfile=None, remotes=None,
                   update=None, check_update=False):
        """ Compute the dependency graph, starting from a root package, evaluation the graph with
        the provided configuration in profile_build, and profile_host. The resulting graph is a
        graph of recipes, but packages are not computed yet (package_ids) will be empty in the
        result. The result might have errors, like version or configuration conflicts, but it is
        still possible to inspect it. Only trying to install such graph will fail

        :param root_node: the starting point, an already initialized Node structure, as
            returned by the "load_root_node" api
        :param profile_host: The host profile
        :param profile_build: The build profile
        :param lockfile: A valid lockfile (None by default, means no locked)
        :param remotes: list of remotes we want to check
        :param update: (False by default), if Conan should look for newer versions or
            revisions for already existing recipes in the Conan cache
        :param check_update: For "graph info" command, check if there are recipe updates
        """
        ConanOutput().title("Computing dependency graph")
        app = ConanApp(self._conan_api)

        assert profile_host is not None
        assert profile_build is not None

        remotes = remotes or []
        builder = DepsGraphBuilder(app.proxy, app.loader, app.range_resolver, app.cache, remotes,
                                   update, check_update, self._conan_api._api_helpers.global_conf)
        deps_graph = builder.load_graph(root_node, profile_host, profile_build, lockfile)
        return deps_graph

    def analyze_binaries(self, graph, build_mode=None, remotes=None, update=None,
                         lockfile=None, build_modes_test=None, tested_graph=None):
        """ Given a dependency graph, will compute the package_ids of all recipes in the graph, and
        evaluate if they should be built from sources, downloaded from a remote server, of if the
        packages are already in the local Conan cache

        :param lockfile:
        :param graph: a Conan dependency graph, as returned by "load_graph()"
        :param build_mode: TODO: Discuss if this should be a BuildMode object or list of arguments
        :param remotes: list of remotes
        :param update: (``False`` by default), if Conan should look for newer versions or
            revisions for already existing recipes in the Conan cache. It also accepts an array of
            reference patterns to limit the update to those references if any of the items match.
            Eg. ``False``, ``None`` or ``[]`` *means no update*,
            ``True`` or ``["*"]`` *means update all*,
            and ``["pkgA/*", "pkgB/1.0@user/channel"]`` *means to update only specific packages*.
        :param build_modes_test: the --build-test argument
        :param tested_graph: In case of a "test_package", the graph being tested
        """
        ConanOutput().title("Computing necessary packages")
        conan_app = ConanBasicApp(self._conan_api)
        binaries_analyzer = GraphBinariesAnalyzer(conan_app, self._conan_api._api_helpers.global_conf,
                                                  self._helpers.hook_manager)
        binaries_analyzer.evaluate_graph(graph, build_mode, lockfile, remotes, update,
                                         build_modes_test, tested_graph)

    @staticmethod
    def find_first_missing_binary(graph, missing=None):
        """ (Experimental) Given a dependency graph, will return the first node with a
        missing binary package
        """
        for node in graph.ordered_iterate():
            if ((not missing and node.binary == BINARY_MISSING)  # First missing binary or specified
                    or (missing and ref_matches(node.ref, missing, is_consumer=None))):
                return node.ref, node.conanfile.info
        raise ConanException("There is no missing binary")

    @staticmethod
    def build_order(deps_graph, order_by="recipe", reduce=False, profile_args=None):
        install_graph = InstallGraph(deps_graph, order_by=order_by,
                                     profile_args=ProfileArgs.from_args(profile_args))
        if reduce:
            if order_by is None:
                raise ConanException("--reduce needs --order-by argument defined")
            install_graph.reduce()
        return install_graph

    @staticmethod
    def build_order_merge(files, reduce=False):
        result = InstallGraph.load(files[0])
        if result.reduced:
            raise ConanException(f"Reduced build-order file cannot be merged: {files[0]}")
        for f in files[1:]:
            install_graph = InstallGraph.load(f)
            if install_graph.reduced:
                raise ConanException(f"Reduced build-order file cannot be merged: {f}")
            result.merge(install_graph)
        if reduce:
            result.reduce()
        return result
