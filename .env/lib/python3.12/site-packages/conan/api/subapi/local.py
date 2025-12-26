import os

from conan.cli import make_abs_path
from conan.internal.conan_app import ConanApp
from conan.internal.api.local.editable import EditablePackages
from conan.internal.methods import run_build_method, run_source_method
from conan.internal.graph.graph import CONTEXT_HOST
from conan.internal.graph.profile_node_definer import initialize_conanfile_profile
from conan.internal.errors import conanfile_exception_formatter
from conan.errors import ConanException
from conan.api.model import RecipeReference
from conan.internal.util.files import chdir


class LocalAPI:

    def __init__(self, conan_api, helpers):
        self._conan_api = conan_api
        self._helpers = helpers
        self.editable_packages = EditablePackages(conan_api.home_folder)

    @staticmethod
    def get_conanfile_path(path, cwd, py):
        """
        param py= True: Must be .py, None: Try .py, then .txt
        """
        path = make_abs_path(path, cwd)

        if os.path.isdir(path):  # Can be a folder
            path_py = os.path.join(path, "conanfile.py")
            if py:
                path = path_py
            else:
                path_txt = os.path.join(path, "conanfile.txt")
                if os.path.isfile(path_py) and os.path.isfile(path_txt):
                    raise ConanException("Ambiguous command, both conanfile.py and conanfile.txt exist")
                path = path_py if os.path.isfile(path_py) else path_txt

        if not os.path.isfile(path):  # Must exist
            raise ConanException("Conanfile not found at %s" % path)

        if py and not path.endswith(".py"):
            raise ConanException("A conanfile.py is needed, " + path + " is not acceptable")

        return path

    def editable_add(self, path, name=None, version=None, user=None, channel=None, cwd=None,
                     output_folder=None, remotes=None):
        path = self.get_conanfile_path(path, cwd, py=True)
        app = ConanApp(self._conan_api)
        conanfile = app.loader.load_named(path, name, version, user, channel, remotes=remotes)
        if conanfile.name is None or conanfile.version is None:
            raise ConanException("Editable package recipe should declare its name and version")
        ref = RecipeReference(conanfile.name, conanfile.version, conanfile.user, conanfile.channel)
        # Retrieve conanfile.py from target_path
        target_path = self.get_conanfile_path(path=path, cwd=cwd, py=True)
        output_folder = make_abs_path(output_folder) if output_folder else None
        # Check the conanfile is there, and name/version matches
        self.editable_packages.add(ref, target_path, output_folder=output_folder)
        return ref

    def editable_remove(self, path=None, requires=None, cwd=None):
        if path:
            path = make_abs_path(path, cwd)
            path = os.path.join(path, "conanfile.py")
        return self.editable_packages.remove(path, requires)

    def editable_list(self):
        return self.editable_packages.edited_refs

    def source(self, path, name=None, version=None, user=None, channel=None, remotes=None):
        """ calls the 'source()' method of the current (user folder) conanfile.py
        """
        app = ConanApp(self._conan_api)
        conanfile = app.loader.load_consumer(path, name=name, version=version,
                                             user=user, channel=channel, graph_lock=None,
                                             remotes=remotes)
        # This profile is empty, but with the conf from global.conf
        profile = self._conan_api.profiles.get_profile([])
        initialize_conanfile_profile(conanfile, profile, profile, CONTEXT_HOST, False)
        # This is important, otherwise the ``conan source`` doesn't define layout and fails
        if hasattr(conanfile, "layout"):
            with conanfile_exception_formatter(conanfile, "layout"):
                conanfile.layout()

        folder = conanfile.recipe_folder if conanfile.folders.root is None else \
            os.path.normpath(os.path.join(conanfile.recipe_folder, conanfile.folders.root))

        conanfile.folders.set_base_source(folder)
        conanfile.folders.set_base_export_sources(folder)
        conanfile.folders.set_base_recipe_metadata(os.path.join(folder, "metadata"))
        # The generators are needed for the "conan source" local case with tool-requires
        conanfile.folders.set_base_generators(folder)
        conanfile.folders.set_base_build(None)
        conanfile.folders.set_base_package(None)

        hook_manager = self._helpers.hook_manager
        run_source_method(conanfile, hook_manager)

    def build(self, conanfile):
        """ calls the 'build()' method of the current (user folder) conanfile.py
        """
        hook_manager = self._helpers.hook_manager
        conanfile.folders.set_base_package(conanfile.folders.base_build)
        conanfile.folders.set_base_pkg_metadata(os.path.join(conanfile.build_folder, "metadata"))
        run_build_method(conanfile, hook_manager)

    @staticmethod
    def test(conanfile):
        """ calls the 'test()' method of the current (user folder) test_package/conanfile.py
        """
        with conanfile_exception_formatter(conanfile, "test"):
            with chdir(conanfile.build_folder):
                conanfile.test()

    def inspect(self, conanfile_path, remotes, lockfile, name=None, version=None, user=None,
                channel=None):
        app = ConanApp(self._conan_api)
        conanfile = app.loader.load_named(conanfile_path, name=name, version=version, user=user,
                                          channel=channel, remotes=remotes, graph_lock=lockfile)
        return conanfile

    def reinit(self):
        self.editable_packages = EditablePackages(self._conan_api.home_folder)
