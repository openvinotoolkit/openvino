import os
import sys
import textwrap
from fnmatch import fnmatch
from io import StringIO

import yaml

from conan.api.model import LOCAL_RECIPES_INDEX
from conan.api.output import ConanOutput
from conan.internal.cache.home_paths import HomePaths
from conan.internal.api.export import cmd_export
from conan.internal.hook_manager import HookManager
from conan.internal.loader import ConanFileLoader
from conan.internal.errors import ConanReferenceDoesNotExistInDB, RecipeNotFoundException, \
    PackageNotFoundException
from conan.errors import ConanException
from conan.internal.model.conf import ConfDefinition
from conan.api.model import RecipeReference
from conan.internal.util.files import load, save, rmdir, copytree_compat


def add_local_recipes_index_remote(home_folder, remote):
    if remote.remote_type != LOCAL_RECIPES_INDEX:
        return
    local_recipes_index_path = HomePaths(home_folder).local_recipes_index_path
    repo_folder = os.path.join(local_recipes_index_path, remote.name)

    output = ConanOutput()
    if os.path.exists(repo_folder):
        output.warning(f"The cache folder for remote {remote.name} existed, removing it")
        rmdir(repo_folder)

    cache_path = os.path.join(repo_folder, ".conan")
    hook_folder = HomePaths(cache_path).hooks_path
    trim_hook = os.path.join(hook_folder, "hook_trim_conandata.py")
    hook_content = textwrap.dedent("""\
        from conan.tools.files import trim_conandata
        def post_export(conanfile):
            if conanfile.conan_data:
                trim_conandata(conanfile)
        """)
    save(trim_hook, hook_content)


def remove_local_recipes_index_remote(home_folder, remote):
    if remote.remote_type == LOCAL_RECIPES_INDEX:
        local_recipes_index_path = HomePaths(home_folder).local_recipes_index_path
        local_recipes_index_path = os.path.join(local_recipes_index_path, remote.name)
        ConanOutput().info(f"Removing temporary files for '{remote.name}' "
                           f"local-recipes-index remote")
        rmdir(local_recipes_index_path)


class RestApiClientLocalRecipesIndex:
    """
    Implements the RestAPI but instead of over HTTP for a remote server, using just
    a local folder assuming the conan-center-index repo layout
    """

    def __init__(self, remote, home_folder):
        self._remote = remote
        local_recipes_index_path = HomePaths(home_folder).local_recipes_index_path
        local_recipes_index_path = os.path.join(local_recipes_index_path, remote.name, ".conan")
        repo_folder = self._remote.url

        from conan.internal.conan_app import LocalRecipesIndexApp
        self._app = LocalRecipesIndexApp(local_recipes_index_path)
        self._hook_manager = HookManager(HomePaths(local_recipes_index_path).hooks_path)
        self._layout = _LocalRecipesIndexLayout(repo_folder)

    def call_method(self, method_name, *args, **kwargs):
        return getattr(self, method_name)(*args, **kwargs)

    def get_recipe(self, ref, dest_folder):
        export_folder = self._app.cache.recipe_layout(ref).export()
        return self._copy_files(export_folder, dest_folder)

    def get_recipe_sources(self, ref, dest_folder):
        try:
            export_sources = self._app.cache.recipe_layout(ref).export_sources()
        except ConanReferenceDoesNotExistInDB:
            # This can happen when there a local-recipes-index is being queried for sources it
            # doesn't contain
            # If that is the case, check if they are in the repo, try to export
            exported_ref = self._export_recipe(ref)  # This will raise NotFound if non existing
            export_sources = self._app.cache.recipe_layout(exported_ref).export_sources()
        return self._copy_files(export_sources, dest_folder)

    def get_package(self, pref, dest_folder, metadata, only_metadata):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support "
                             "binary packages")

    def upload_recipe(self, ref, files_to_upload):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support upload")

    def upload_package(self, pref, files_to_upload):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support upload")

    def authenticate(self, user, password):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support "
                             "authentication")

    def check_credentials(self, force_auth=False):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support upload")

    def search(self, pattern=None):
        return self._layout.get_recipes_references(pattern)

    def search_packages(self, reference, _=False):
        assert self and reference
        return {}

    def remove_recipe(self, ref):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support remove")

    def remove_all_packages(self, ref):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support remove")

    def remove_packages(self, prefs):
        raise ConanException(f"Remote local-recipes-index '{self._remote.name}' doesn't support remove")

    def get_recipe_revisions_references(self, ref):
        ref = self._export_recipe(ref)
        return [ref]

    def get_package_revisions_references(self, pref):
        raise PackageNotFoundException(pref)

    def get_latest_recipe_reference(self, ref):
        ref = self._export_recipe(ref)
        return ref

    def get_latest_package_reference(self, pref, headers):
        raise PackageNotFoundException(pref)

    def get_recipe_revision_reference(self, ref):
        new_ref = self._export_recipe(ref)
        return new_ref

    def get_package_revision_reference(self, pref):
        raise PackageNotFoundException(pref)

    # Helper methods to implement the interface
    def _export_recipe(self, ref):
        folder = self._layout.get_recipe_folder(ref)
        conanfile_path = os.path.join(folder, "conanfile.py")
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            global_conf = ConfDefinition()
            new_ref, _ = cmd_export(self._app.loader, self._app.cache, self._hook_manager,
                                    global_conf, conanfile_path,
                                    ref.name, str(ref.version), None, None, remotes=[self._remote])
        except Exception as e:
            raise ConanException(f"Error while exporting recipe from remote: {self._remote.name}\n"
                                 f"{str(e)}")
        finally:
            export_err = sys.stderr.getvalue()
            sys.stderr = original_stderr
            ConanOutput(scope="local-recipes-index").debug(f"Internal export for {ref}:\n"
                                                           f"{textwrap.indent(export_err, '    ')}")
        if new_ref.user != ref.user or new_ref.channel != ref.channel:
            raise RecipeNotFoundException(ref)
        if ref.revision is not None and new_ref.revision != ref.revision:
            ConanOutput().warning(f"A specific revision '{ref.repr_notime()}' was requested, but it "
                                  "doesn't match the current available revision in source. The "
                                  "local-recipes-index can't provide a specific revision")
            raise RecipeNotFoundException(ref)
        return new_ref

    @staticmethod
    def _copy_files(source_folder, dest_folder):
        if not os.path.exists(source_folder):
            return {}
        copytree_compat(source_folder, dest_folder)
        ret = {}
        for root, _, _files in os.walk(dest_folder):
            for _f in _files:
                rel = os.path.relpath(os.path.join(root, _f), dest_folder)
                ret[rel] = os.path.join(dest_folder, root, _f)
        return ret


class _LocalRecipesIndexLayout:

    def __init__(self, base_folder):
        self._base_folder = base_folder

    def _get_base_folder(self, recipe_name):
        return os.path.join(self._base_folder, "recipes", recipe_name)

    @staticmethod
    def _load_config_yml(folder):
        config = os.path.join(folder, "config.yml")
        if not os.path.isfile(config):
            return None
        return yaml.safe_load(load(config))

    def get_recipes_references(self, pattern):
        name_pattern = pattern.split("/", 1)[0]
        recipes_dir = os.path.join(self._base_folder, "recipes")
        if not os.path.isdir(recipes_dir):
            raise ConnectionError("Cannot connect to 'local-recipes-index' repository, missing "
                                  f"'recipes' folder: {recipes_dir}")
        recipes = os.listdir(recipes_dir)
        recipes.sort()
        ret = []
        excluded = set()

        original_pattern = pattern
        try:
            pattern_ref = RecipeReference.loads(pattern)
            # We don't care about user/channel for now, we'll check for those below,
            # just add all candidates for now
            pattern = f"{pattern_ref.name}/{pattern_ref.version}"
        except:
            # pattern = pattern
            pattern = f"{name_pattern}/*"
            original_pattern = pattern
            pass

        loader = ConanFileLoader(None)
        for r in recipes:
            if r.startswith("."):
                # Skip hidden folders, no recipes should start with a dot
                continue
            if not fnmatch(r, name_pattern):
                continue
            folder = self._get_base_folder(r)
            config_yml = self._load_config_yml(folder)
            if config_yml is None:
                raise ConanException(f"Corrupted repo, folder {r} without 'config.yml'")
            versions = config_yml["versions"]
            for v in versions:
                # TODO: Check the search pattern is the same as remotes and cache
                ref = f"{r}/{v}"
                if not fnmatch(ref, pattern):
                    continue
                subfolder = versions[v]["folder"]
                # This check can be removed after compatibility with 2.0
                conanfile = os.path.join(recipes_dir, r, subfolder, "conanfile.py")
                conanfile_content = load(conanfile)

                if "from conans" in conanfile_content or "import conans" in conanfile_content:
                    excluded.add(r)
                    continue
                ref = RecipeReference.loads(ref)
                try:
                    recipe = loader.load_basic(conanfile)
                    ref.user = recipe.user
                    ref.channel = recipe.channel
                except Exception as e:
                    ConanOutput().warning(f"Couldn't load recipe {conanfile}: {e}")
                if ref.matches(original_pattern, None):
                    # Check again the pattern with the user/channel
                    ret.append(ref)
        if excluded:
            ConanOutput().warning(f"Excluding recipes not Conan 2.0 ready: {', '.join(excluded)}")
        return ret

    def get_recipe_folder(self, ref):
        folder = self._get_base_folder(ref.name)
        data = self._load_config_yml(folder)
        if data is None:
            raise RecipeNotFoundException(ref)
        versions = data["versions"]
        if str(ref.version) not in versions:
            raise RecipeNotFoundException(ref)
        subfolder = versions[str(ref.version)]["folder"]
        return os.path.join(folder, subfolder)
