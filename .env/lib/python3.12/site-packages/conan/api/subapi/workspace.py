import inspect
import os
import shutil
import textwrap
from pathlib import Path

from conan import ConanFile
from conan.api.model import RecipeReference
from conan.api.output import ConanOutput
from conan.cli import make_abs_path
from conan.cli.printers.graph import print_graph_basic, print_graph_packages
from conan.errors import ConanException
from conan.internal.conan_app import ConanApp
from conan.internal.errors import conanfile_exception_formatter
from conan.internal.graph.install_graph import ProfileArgs
from conan.internal.methods import auto_language, auto_shared_fpic_config_options, \
    auto_shared_fpic_configure
from conan.internal.model.options import Options
from conan.internal.model.workspace import Workspace, WORKSPACE_YML, WORKSPACE_PY, WORKSPACE_FOLDER
from conan.tools.scm import Git
from conan.internal.graph.graph import (RECIPE_EDITABLE, DepsGraph, CONTEXT_HOST, RECIPE_VIRTUAL,
                                        Node, RECIPE_CONSUMER)
from conan.internal.graph.graph import TransitiveRequirement
from conan.internal.graph.profile_node_definer import consumer_definer, initialize_conanfile_profile
from conan.internal.loader import load_python_file
from conan.internal.source import retrieve_exports_sources
from conan.internal.util.files import merge_directories, save


def _find_ws_folder():
    path = Path(os.getcwd())
    while path.is_dir() and len(path.parts) > 1:  # finish at '/' or 'conanws/'
        if path.name == WORKSPACE_FOLDER:
            if (path / WORKSPACE_YML).is_file() or (path / WORKSPACE_PY).is_file():
                return str(path)
        if (path / WORKSPACE_YML).is_file() or (path / WORKSPACE_PY).is_file():
            return str(path)
        else:
            path = path.parent


def _load_workspace(ws_folder, conan_api):
    """ loads a conanfile basic object without evaluating anything, returns the module too
    """
    wspy = os.path.join(ws_folder, WORKSPACE_PY)
    if not os.path.isfile(wspy):
        ConanOutput().info(f"{WORKSPACE_PY} doesn't exist in {ws_folder}, using default behavior")
        assert os.path.exists(os.path.join(ws_folder, WORKSPACE_YML))
        ws = Workspace(ws_folder, conan_api)
    else:
        try:
            module, module_id = load_python_file(wspy)
            ws = _parse_module(module, module_id)
            ws = ws(ws_folder, conan_api)
        except ConanException as e:
            raise ConanException(f"Error loading {WORKSPACE_PY} at '{wspy}': {e}")
    return ws


def _parse_module(conanfile_module, module_id):
    result = None
    for name, attr in conanfile_module.__dict__.items():
        if (name.startswith("_") or not inspect.isclass(attr) or
                attr.__dict__.get("__module__") != module_id):
            continue

        if issubclass(attr, Workspace) and attr != Workspace:
            if result is None:
                result = attr
            else:
                raise ConanException("More than 1 Workspace in the file")

    if result is None:
        raise ConanException("No subclass of Workspace")

    return result


class WorkspaceAPI:
    TEST_ENABLED = False

    def __init__(self, conan_api):
        self._enabled = True
        self._conan_api = conan_api
        self._folder = _find_ws_folder()
        if self._folder:
            ConanOutput().warning(f"Workspace found: {self._folder}")
            if (WorkspaceAPI.TEST_ENABLED or os.getenv("CONAN_WORKSPACE_ENABLE")) != "will_break_next":
                ConanOutput().warning("Workspace ignored as CONAN_WORKSPACE_ENABLE is not set")
                self._folder = None
            else:
                ConanOutput().warning(f"Workspace is a dev-only feature, exclusively for testing")
                self._ws = _load_workspace(self._folder, conan_api)  # Error if not loading

    def enable(self, value):
        self._enabled = value

    def name(self):
        self._check_ws()
        return self._ws.name()

    def folder(self):
        """
        @return: the current workspace folder where the conanws.yml or conanws.py is located
        """
        self._check_ws()
        return self._folder

    def packages(self):
        """
        @return: Returns {RecipeReference: {"path": full abs-path, "output_folder": abs-path}}
        """
        if not self._folder or not self._enabled:
            return
        packages = {}
        for editable_info in self._ws.packages():
            rel_path = editable_info["path"]
            path = os.path.normpath(os.path.join(self._folder, rel_path, "conanfile.py"))
            if not os.path.isfile(path):
                raise ConanException(f"Workspace package not found: {path}")
            ref = editable_info.get("ref")
            try:
                if ref is None:
                    conanfile = self._ws.load_conanfile(rel_path)
                    reference = RecipeReference(name=conanfile.name, version=conanfile.version,
                                                user=conanfile.user, channel=conanfile.channel)
                else:
                    reference = RecipeReference.loads(ref)
                reference.validate_ref(reference)
            except Exception as e:
                raise ConanException(f"Workspace package reference could not be deduced by"
                                     f" {rel_path}/conanfile.py or it is not"
                                     f" correctly defined in the conanws.yml file: {e}")
            if reference in packages:
                raise ConanException(f"Workspace package '{str(reference)}' already exists.")
            packages[reference] = {"path": path}
            if editable_info.get("output_folder"):
                packages[reference]["output_folder"] = (
                    os.path.normpath(os.path.join(self._folder, editable_info["output_folder"]))
                )
        return packages

    def open(self, ref, remotes, cwd=None):
        cwd = cwd or os.getcwd()
        app = ConanApp(self._conan_api)
        ref = RecipeReference.loads(ref) if isinstance(ref, str) else ref
        recipe = app.proxy.get_recipe(ref, remotes, update=False, check_update=False)

        layout, recipe_status, remote = recipe
        if recipe_status == RECIPE_EDITABLE:
            raise ConanException(f"Can't open a dependency that is already an editable: {ref}")
        ref = layout.reference
        conanfile_path = layout.conanfile()
        conanfile, module = app.loader.load_basic_module(conanfile_path, remotes=remotes)

        scm = conanfile.conan_data.get("scm") if conanfile.conan_data else None
        dst_path = os.path.join(cwd, ref.name)
        if scm is None:
            conanfile.output.warning("conandata doesn't contain 'scm' information\n"
                                     "doing a local copy!!!")
            shutil.copytree(layout.export(), dst_path)
            retrieve_exports_sources(app.remote_manager, layout, conanfile, ref, remotes)
            export_sources = layout.export_sources()
            if os.path.exists(export_sources):
                conanfile.output.warning("There are export-sources, copying them, but the location"
                                         " might be incorrect, use 'scm' approach")
                merge_directories(export_sources, dst_path)
        else:
            git = Git(conanfile, folder=cwd)
            git.clone(url=scm["url"], target=ref.name)
            git.folder = ref.name  # change to the cloned folder
            git.checkout(commit=scm["commit"])
        return dst_path

    def _check_ws(self):
        if not self._folder:
            raise ConanException(f"Workspace not defined, please create a "
                                 f"'{WORKSPACE_PY}' or '{WORKSPACE_YML}' file")

    def add(self, path, name=None, version=None, user=None, channel=None, cwd=None,
            output_folder=None, remotes=None):
        """
        Add a new editable package to the current workspace (the current workspace must exist)
        @param path: The path to the folder containing the conanfile.py that defines the package
        @param name: (optional) The name of the package to be added if not defined in recipe
        @param version:
        @param user:
        @param channel:
        @param cwd:
        @param output_folder:
        @param remotes:
        @return: The reference of the added package
        """
        self._check_ws()
        full_path = self._conan_api.local.get_conanfile_path(path, cwd, py=True)
        app = ConanApp(self._conan_api)
        conanfile = app.loader.load_named(full_path, name, version, user, channel, remotes=remotes)
        if conanfile.name is None or conanfile.version is None:
            raise ConanException("Editable package recipe should declare its name and version")
        ref = RecipeReference(conanfile.name, conanfile.version, conanfile.user, conanfile.channel)
        ref.validate_ref()
        output_folder = make_abs_path(output_folder) if output_folder else None
        # Check the conanfile is there, and name/version matches
        self._ws.add(ref, full_path, output_folder)
        return ref

    def complete(self, profile_host, profile_build, lockfile, remotes, update):
        packages = self.packages()
        if not packages:
            ConanOutput().info("There are no packages in this workspace, nothing to complete")
            return

        for ref, info in packages.items():
            ConanOutput().title(f"Computing the dependency graph for package: {ref}")
            gapi = self._conan_api.graph
            deps_graph = gapi.load_graph_requires([ref], None, profile_host, profile_build,
                                                  lockfile, remotes, update)
            deps_graph.report_graph_error()
            print_graph_basic(deps_graph)

            nodes_to_complete = []
            for node in deps_graph.nodes[1:]:  # Exclude the current virtual root
                if node.recipe != RECIPE_EDITABLE:
                    # sanity check, a pacakge in the cache cannot have dependencies to the workspace
                    if any(d.node.recipe == RECIPE_EDITABLE for d in node.transitive_deps.values()):
                        nodes_to_complete.append(node)

            if not nodes_to_complete:
                ConanOutput().info("There are no intermediate packages to add to the workspace")
                return

            for node in nodes_to_complete:
                full_path = os.path.join(self._folder, node.name, "conanfile.py")
                dep_ref = node.ref
                ConanOutput().info(f"Adding to workspace {dep_ref}")
                try:
                    self._ws.add(dep_ref, full_path, output_folder=None)
                except ConanException:
                    if os.path.isfile(full_path):
                        raise
                    ConanOutput().info(f"Conanfile in {node.name} not found, trying "
                                       "to open it first")
                    self.open(dep_ref, remotes, cwd=self._folder)
                    self._ws.add(dep_ref, full_path, output_folder=None)

    @staticmethod
    def init(path):
        abs_path = make_abs_path(path)
        os.makedirs(abs_path, exist_ok=True)
        ws_yml_file = Path(abs_path, WORKSPACE_YML)
        ws_py_file = Path(abs_path, WORKSPACE_PY)
        if not ws_yml_file.exists():
            ConanOutput().success(f"Created empty {WORKSPACE_YML} in {path}")
            save(ws_yml_file, "")
        if not ws_py_file.exists():
            ConanOutput().success(f"Created minimal {WORKSPACE_PY} in {path}")
            ws_name = os.path.basename(abs_path)
            save(ws_py_file, textwrap.dedent(f'''\
            from conan import Workspace

            class MyWorkspace(Workspace):
               """
               Minimal Workspace class definition.
               More info: https://docs.conan.io/2/incubating.html#workspaces
               """
               def name(self):
                  return "{ws_name}"
            '''))

    def remove(self, path):
        self._check_ws()
        return self._ws.remove(path)

    def clean(self):
        self._check_ws()
        return self._ws.clean()

    def info(self):
        self._check_ws()
        return {"name": self._ws.name(),
                "folder": self._folder,
                "packages": self._ws.packages()}

    @staticmethod
    def _init_options(conanfile, options):
        if hasattr(conanfile, "config_options"):
            with conanfile_exception_formatter(conanfile, "config_options"):
                conanfile.config_options()
        elif "auto_shared_fpic" in conanfile.implements:
            auto_shared_fpic_config_options(conanfile)

        auto_language(conanfile)  # default implementation removes `compiler.cstd`

        # Assign only the current package options values, but none of the dependencies
        conanfile.options.apply_downstream(Options(), options, None, True)

        if hasattr(conanfile, "configure"):
            with conanfile_exception_formatter(conanfile, "configure"):
                conanfile.configure()
        elif "auto_shared_fpic" in conanfile.implements:
            auto_shared_fpic_configure(conanfile)

    def super_build_graph(self, deps_graph, profile_host, profile_build):
        order = []
        packages = self.packages()

        def find_folder(ref):
            return next(os.path.dirname(os.path.relpath(p["path"], self._folder)) for p_ref, p in
                        packages.items() if p_ref == ref)

        for level in deps_graph.by_levels():
            items = [item for item in level if item.recipe == "Editable"]
            level_order = []
            for node in items:
                conanfile = node.conanfile
                if hasattr(conanfile, "layout"):
                    with conanfile_exception_formatter(conanfile, "layout"):
                        conanfile.layout()
                base_folder = find_folder(node.ref)
                src_folder = os.path.normpath(os.path.join(base_folder, conanfile.folders.source))
                level_order.append({"ref": node.ref, "folder": src_folder.replace("\\", "/")})
            order.append(level_order)

        self._ws.build_order(order)

        ConanOutput().title("Collapsing workspace packages")

        root_class = self._ws.root_conanfile()
        if root_class is not None:
            conanfile = root_class(f"{WORKSPACE_PY} base project Conanfile")
            # To inject things like cmd_wrapper to the consumer conanfile, so self.run() works
            helpers = ConanApp(self._conan_api).loader._conanfile_helpers  # noqa
            conanfile._conan_helpers = helpers
            conanfile._conan_is_consumer = True
            initialize_conanfile_profile(conanfile, profile_build, profile_host, CONTEXT_HOST,
                                         is_build_require=False)
            # consumer_definer(conanfile, profile_host, profile_build)
            self._init_options(conanfile, profile_host.options)
            for field in ("requires", "build_requires", "test_requires", "requirements", "build",
                          "source", "package"):
                if getattr(conanfile, field, None):
                    raise ConanException(f"Conanfile in conanws.py shouldn't have '{field}'")
            root = Node(None, conanfile, context=CONTEXT_HOST, recipe=RECIPE_CONSUMER,
                        path=self._folder)  # path lets use the conanws.py folder
            root.should_build = True  # It is a consumer, this is something we are building
        else:
            ConanOutput().info(f"Workspace {WORKSPACE_PY} not found in the workspace folder, "
                               "using default behavior")
            conanfile = ConanFile(display_name="cli")
            consumer_definer(conanfile, profile_host, profile_build)
            root = Node(ref=None, conanfile=conanfile, context=CONTEXT_HOST, recipe=RECIPE_VIRTUAL)

        result = DepsGraph()  # TODO: We might need to copy more information from the original graph
        result.add_node(root)
        conanfile.workspace_packages = {}

        self._check_graph(deps_graph)
        for node in deps_graph.nodes[1:]:  # Exclude the current root
            if node.recipe != RECIPE_EDITABLE:
                result.add_node(node)
                continue
            # At the moment we are exposing the full conanfile, docs will warn against usage of
            # non pure functions
            conanfile.workspace_packages[node.ref] = node.conanfile
            for r, t in node.transitive_deps.items():
                if t.node.recipe == RECIPE_EDITABLE:
                    continue
                existing = root.transitive_deps.pop(r, None)
                if existing is None:
                    root.transitive_deps[r] = t
                else:
                    require = existing.require
                    require.aggregate(r)
                    root.transitive_deps[require] = TransitiveRequirement(require, t.node)

        # The graph edges must be defined too
        for r, t in root.transitive_deps.items():
            result.add_edge(root, t.node, r)

        return result

    @staticmethod
    def _check_graph(graph):
        for node in graph.nodes[1:]:  # Exclude the current root
            if node.recipe != RECIPE_EDITABLE:
                # sanity check, a pacakge in the cache cannot have dependencies to the workspace
                deps_edit = [d.node for d in node.transitive_deps.values()
                             if d.node.recipe == RECIPE_EDITABLE]
                if deps_edit:
                    raise ConanException(f"Workspace definition error. Package {node} in the "
                                         f"Conan cache has dependencies to packages "
                                         f"in the workspace: {deps_edit}\n"
                                         "Try the 'conan workspace complete' to open/add "
                                         "intermediate packages")

    def export(self, lockfile=None, remotes=None):
        self._check_ws()
        exported = []
        for ref, info in self.packages().items():
            exported_ref = self._conan_api.export.export(info["path"], ref.name, str(ref.version),
                                                         ref.user, ref.channel,
                                                         lockfile=lockfile, remotes=remotes)
            ref, _ = exported_ref
            exported.append(ref)
        return exported

    def select_packages(self, packages):
        self._check_ws()
        editable = self.packages()
        packages = packages or []
        selected_editables = {}
        for ref, info in editable.items():
            if packages and not any(ref.matches(p, False) for p in packages):
                continue
            selected_editables[ref] = info
        if not selected_editables:
            raise ConanException("There are no selected packages defined in the workspace")

        return selected_editables

    def build_order(self, packages, profile_host, profile_build, build_mode, lockfile, remotes,
                    profile_args, update=False):
        ConanOutput().title(f"Computing dependency graph for each package")
        conan_api = self._conan_api
        from conan.internal.graph.install_graph import InstallGraph
        install_order = InstallGraph(None)

        for ref, info in packages.items():
            ConanOutput().title(f"Computing the dependency graph for package: {ref}")

            deps_graph = conan_api.graph.load_graph_requires([ref], None,
                                                             profile_host, profile_build, lockfile,
                                                             remotes, update)
            deps_graph.report_graph_error()
            print_graph_basic(deps_graph)

            self._check_graph(deps_graph)

            conan_api.graph.analyze_binaries(deps_graph, build_mode, remotes=remotes, update=update,
                                             lockfile=lockfile)
            print_graph_packages(deps_graph)

            ConanOutput().success(f"\nAggregating build-order for package: {ref}")
            install_graph = InstallGraph(deps_graph, order_by="recipe",
                                         profile_args=ProfileArgs.from_args(profile_args))
            install_graph.raise_errors()
            install_order.merge(install_graph)

        return install_order
