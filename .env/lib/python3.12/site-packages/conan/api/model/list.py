import copy
import fnmatch
import json
import os
from json import JSONDecodeError
from typing import Iterable, Tuple, Dict

from conan.api.model import RecipeReference, PkgReference
from conan.api.output import ConanOutput
from conan.errors import ConanException
from conan.internal.errors import NotFoundException
from conan.internal.model.version_range import VersionRange
from conan.internal.graph.graph import RECIPE_EDITABLE, RECIPE_CONSUMER, RECIPE_PLATFORM, \
    RECIPE_VIRTUAL, BINARY_SKIP, BINARY_MISSING, BINARY_INVALID
from conan.internal.util.files import load


class MultiPackagesList:
    """ A collection of PackagesList by remote name."""
    def __init__(self):
        self.lists = {}

    def __getitem__(self, name):
        try:
            return self.lists[name]
        except KeyError:
            raise ConanException(f"'{name}' doesn't exist in package list")

    def add(self, name, pkg_list):
        self.lists[name] = pkg_list

    def add_error(self, remote_name, error):
        self.lists[remote_name] = {"error": error}

    def serialize(self):
        """ Serialize object to a dictionary."""
        return {k: v.serialize() if isinstance(v, PackagesList) else v
                for k, v in self.lists.items()}

    def merge(self, other):
        for k, v in other.lists.items():
            self.lists.setdefault(k, PackagesList()).merge(v)

    def keep_outer(self, other):
        for namespace, other_pkg_list in other.lists.items():
            self.lists.get(namespace, PackagesList()).keep_outer(other_pkg_list)

    @staticmethod
    def load(file):
        """ Create an instance of the class from a serialized JSON file path pointed by ``file``."""
        try:
            content = json.loads(load(file))
        except JSONDecodeError as e:
            raise ConanException(f"Package list file invalid JSON: {file}\n{e}")
        except Exception as e:
            raise ConanException(f"Package list file missing or broken: {file}\n{e}")
        # Check if input json is not a graph file
        if "graph" in content:
            base_path = os.path.basename(file)
            raise ConanException(
                'Expected a package list file but found a graph file. You can create a "package list" JSON file by running:\n\n'
                f"\tconan list --graph {base_path} --format=json > pkglist.json\n\n"
                "More Info at 'https://docs.conan.io/2/examples/commands/pkglists.html"
            )
        result = {}
        for remote, pkglist in content.items():
            if "error" in pkglist:
                result[remote] = pkglist
            else:
                result[remote] = PackagesList.deserialize(pkglist)
        pkglist = MultiPackagesList()
        pkglist.lists = result
        return pkglist

    @staticmethod
    def load_graph(graphfile, graph_recipes=None, graph_binaries=None, context=None):
        """ Create an instance of the class from a graph file path, which is
        the json format returned by a few commands
        like ``conan graph info`` or ``conan create/install.``

        :parameter str graphfile: Path to the graph file
        :parameter list[str] graph_recipes: List for kinds of recipes to return.
            For example ``"cache"`` will return only recipes in the local cache,
            ``"download"`` will return only recipes that have been downloaded,
            and passing ``"*"`` will return all recipes.
        :parameter list[str] graph_binaries: List for kinds of binaries to return.
            For example ``"cache"`` will return only binaries in the local cache,
            ``"download"`` will return only binaries that have been downloaded,
            ``"build"`` will return only binaries that are built,
            ``"missing"`` will return only binaries that are missing,
            ``"invalid"`` will return only binaries that are invalid,
            and passing ``"*"`` will return all binaries.
        :parameter str context: Context to filter the graph,
            can be ``"host"``, ``"build"``, ``"host-only"`` or ``"build-only"``
        """
        if not os.path.isfile(graphfile):
            raise ConanException(f"Graph file not found: {graphfile}")
        try:
            base_context = context.split("-")[0] if context else None
            graph = json.loads(load(graphfile))
            # Check if input json is a graph file
            if "graph" not in graph:
                raise ConanException(
                    'Expected a graph file but found an unexpected JSON file format. You can create a "graph" JSON file by running:\n\n'
                    f"\tconan [ graph-info | create | export-pkg | install | test ] --format=json > graph.json\n\n"
                    "More Info at 'https://docs.conan.io/2/reference/commands/formatters/graph_info_json_formatter.html"
                )

            mpkglist = MultiPackagesList._define_graph(graph, graph_recipes, graph_binaries,
                                                       context=base_context)
            if context == "build-only":
                host = MultiPackagesList._define_graph(graph, graph_recipes, graph_binaries,
                                                       context="host")
                mpkglist.keep_outer(host)
            elif context == "host-only":
                build = MultiPackagesList._define_graph(graph, graph_recipes, graph_binaries,
                                                        context="build")
                mpkglist.keep_outer(build)
            return mpkglist
        except JSONDecodeError as e:
            raise ConanException(f"Graph file invalid JSON: {graphfile}\n{e}")
        except KeyError as e:
            raise ConanException(f'Graph file {graphfile} is missing the required "{e}" key in its contents.\n'
                                 "Note that the graph file should not be filtered "
                                 "if you expect to use it with the list command.")
        except ConanException as e:
            raise e
        except Exception as e:
            raise ConanException(f"Graph file broken: {graphfile}\n{e}")

    @staticmethod
    def _define_graph(graph, graph_recipes=None, graph_binaries=None, context=None):
        pkglist = MultiPackagesList()
        cache_list = PackagesList()
        if graph_recipes is None and graph_binaries is None:
            recipes = ["*"]
            binaries = ["*"]
        else:
            recipes = [r.lower() for r in graph_recipes or []]
            binaries = [b.lower() for b in graph_binaries or []]

        pkglist.lists["Local Cache"] = cache_list
        for node in graph["graph"]["nodes"].values():
            if context and node['context'] != context:
                continue

            # We need to add the python_requires too
            python_requires = node.get("python_requires")
            if python_requires is not None:
                for pyref, pyreq in python_requires.items():
                    pyrecipe = pyreq["recipe"]
                    if pyrecipe == RECIPE_EDITABLE:
                        continue
                    pyref = RecipeReference.loads(pyref)
                    if any(r == "*" or r == pyrecipe for r in recipes):
                        cache_list.add_ref(pyref)
                    pyremote = pyreq["remote"]
                    if pyremote:
                        remote_list = pkglist.lists.setdefault(pyremote, PackagesList())
                        remote_list.add_ref(pyref)

            recipe = node["recipe"]
            if recipe in (RECIPE_EDITABLE, RECIPE_CONSUMER, RECIPE_VIRTUAL, RECIPE_PLATFORM):
                continue

            ref = node["ref"]
            ref = RecipeReference.loads(ref)
            ref.timestamp = node["rrev_timestamp"]
            recipe = recipe.lower()
            if any(r == "*" or r == recipe for r in recipes):
                cache_list.add_ref(ref)

            remote = node["remote"]
            if remote:
                remote_list = pkglist.lists.setdefault(remote, PackagesList())
                remote_list.add_ref(ref)
            pref = PkgReference(ref, node["package_id"], node["prev"], node["prev_timestamp"])
            binary_remote = node["binary_remote"]
            if binary_remote:
                remote_list = pkglist.lists.setdefault(binary_remote, PackagesList())
                remote_list.add_ref(ref)  # Binary listed forces recipe listed
                remote_list.add_pref(pref)

            binary = node["binary"]
            if binary in (BINARY_SKIP, BINARY_INVALID, BINARY_MISSING):
                continue

            binary = binary.lower()
            if any(b == "*" or b == binary for b in binaries):
                cache_list.add_ref(ref)  # Binary listed forces recipe listed
                cache_list.add_pref(pref, node["info"])
        return pkglist


class PackagesList:
    """ A collection of recipes, revisions and packages."""
    def __init__(self):
        self._data = {}

    def __bool__(self):
        """ Whether the package list contains any recipe"""
        return bool(self._data)

    def merge(self, other):
        assert isinstance(other, PackagesList)

        def recursive_dict_update(d, u):  # TODO: repeated from conandata.py
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_dict_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        recursive_dict_update(self._data, other._data)

    def keep_outer(self, other):
        assert isinstance(other, PackagesList)
        if not self._data:
            return

        for ref, info in other._data.items():
            if self._data.get(ref, {}) == info:
                self._data.pop(ref)

    def split(self):
        """
        Returns a list of PackageList, split one per reference.
        This can be useful to parallelize things like upload, parallelizing per-reference
        """
        result = []
        for r, content in self._data.items():
            subpkglist = PackagesList()
            subpkglist._data[r] = content
            result.append(subpkglist)
        return result

    def only_recipes(self) -> None:
        """ Filter out all the packages and package revisions, keep only the recipes and
            recipe revisions in self._data.
        """
        for ref, ref_dict in self._data.items():
            for rrev_dict in ref_dict.get("revisions", {}).values():
                rrev_dict.pop("packages", None)

    def add_refs(self, refs):
        ConanOutput().warning("PackagesLists.add_refs() non-public, non-documented method will be "
                              "removed, use .add_ref() instead", warn_tag="deprecated")
        # RREVS alreday come in ASCENDING order, so upload does older revisions first
        for ref in refs:
            self.add_ref(ref)

    def add_ref(self, ref: RecipeReference) -> None:
        """
        Adds a new RecipeReference to a package list
        """
        ref_dict = self._data.setdefault(str(ref), {})
        if ref.revision:
            revs_dict = ref_dict.setdefault("revisions", {})
            rev_dict = revs_dict.setdefault(ref.revision, {})
            if ref.timestamp:
                rev_dict["timestamp"] = ref.timestamp

    def add_prefs(self, rrev, prefs):
        ConanOutput().warning("PackageLists.add_prefs() non-public, non-documented method will be "
                              "removed, use .add_pref() instead", warn_tag="deprecated")
        # Prevs already come in ASCENDING order, so upload does older revisions first
        for p in prefs:
            self.add_pref(p)

    def add_pref(self, pref: PkgReference, pkg_info: dict = None) -> None:
        """
        Add a PkgReference to an already existing RecipeReference inside a package list
        """
        # Prevs already come in ASCENDING order, so upload does older revisions first
        rev_dict = self.recipe_dict(pref.ref)
        packages_dict = rev_dict.setdefault("packages", {})
        package_dict = packages_dict.setdefault(pref.package_id, {})
        if pref.revision:
            prevs_dict = package_dict.setdefault("revisions", {})
            prev_dict = prevs_dict.setdefault(pref.revision, {})
            if pref.timestamp:
                prev_dict["timestamp"] = pref.timestamp
        if pkg_info is not None:
            package_dict["info"] = pkg_info

    def add_configurations(self, confs):
        ConanOutput().warning("PackageLists.add_configurations() non-public, non-documented method "
                              "will be removed, use .add_pref() instead",
                              warn_tag="deprecated")
        for pref, conf in confs.items():
            rev_dict = self.recipe_dict(pref.ref)
            try:
                rev_dict["packages"][pref.package_id]["info"] = conf
            except KeyError:  # If package_id does not exist, do nothing, only add to existing prefs
                pass

    def refs(self):
        ConanOutput().warning("PackageLists.refs() non-public, non-documented method will be "
                              "removed, use .items() instead", warn_tag="deprecated")
        result = {}
        for ref, ref_dict in self._data.items():
            for rrev, rrev_dict in ref_dict.get("revisions", {}).items():
                t = rrev_dict.get("timestamp")
                recipe = RecipeReference.loads(f"{ref}#{rrev}")  # TODO: optimize this
                if t is not None:
                    recipe.timestamp = t
                result[recipe] = rrev_dict
        return result

    def items(self) -> Iterable[Tuple[RecipeReference, Dict[PkgReference, Dict]]]:
        """ Iterate the contents of the package list.

        The first dictionary is the information directly belonging to the recipe-revision.
        The second dictionary contains PkgReference as keys, and a dictionary with the values
        belonging to that specific package reference (settings, options, etc.).
        """
        for ref, ref_dict in self._data.items():
            for rrev, rrev_dict in ref_dict.get("revisions", {}).items():
                recipe = RecipeReference.loads(f"{ref}#{rrev}")  # TODO: optimize this
                t = rrev_dict.get("timestamp")
                if t is not None:
                    recipe.timestamp = t
                packages = {}
                for package_id, pkg_info in rrev_dict.get("packages", {}).items():
                    prevs = pkg_info.get("revisions", {})
                    for prev, prev_info in prevs.items():
                        t = prev_info.get("timestamp")
                        pref = PkgReference(recipe, package_id, prev, t)
                        packages[pref] = prev_info
                yield recipe, packages

    def recipe_dict(self, ref: RecipeReference):
        """ Gives read/write access to the dictionary containing a specific RecipeReference
        information.
        """
        return self._data[str(ref)]["revisions"][ref.revision]

    def package_dict(self, pref: PkgReference):
        """ Gives read/write access to the dictionary containing a specific PkgReference
        information
        """
        ref_dict = self.recipe_dict(pref.ref)
        return ref_dict["packages"][pref.package_id]["revisions"][pref.revision]

    @staticmethod
    def prefs(ref, recipe_bundle):
        ConanOutput().warning("PackageLists.prefs() non-public, non-documented method will be "
                              "removed, use .items() instead", warn_tag="deprecated")
        result = {}
        for package_id, pkg_bundle in recipe_bundle.get("packages", {}).items():
            prevs = pkg_bundle.get("revisions", {})
            for prev, prev_bundle in prevs.items():
                t = prev_bundle.get("timestamp")
                pref = PkgReference(ref, package_id, prev, t)
                result[pref] = prev_bundle
        return result

    def serialize(self):
        """ Serialize the instance to a dictionary."""
        return copy.deepcopy(self._data)

    @staticmethod
    def deserialize(data):
        """ Loads the data from a serialized dictionary."""
        result = PackagesList()
        result._data = copy.deepcopy(data)
        return result


class ListPattern:
    """ Object holding a pattern that matches recipes, revisions and packages."""

    def __init__(self, expression, rrev="latest", package_id=None, prev="latest", only_recipe=False):
        """
        :param expression: The pattern to match, e.g. ``"name/*:*"``
        :param rrev: The recipe revision to match, defaults to ``"latest"``,
                     can also be ``"!latest"`` or ``"~latest"`` to match all but the latest revision,
                     a pattern like ``"1234*"`` to match a specific revision,
                     or a specific revision like ``"1234"``.
        :param package_id: The package ID to match, defaults to ``None``, which matches all package IDs.
        :param prev: The package revision to match, defaults to ``"latest"``,
                     can also be ``"!latest"`` or ``"~latest"`` to match all but the latest revision,
                     a pattern like ``"1234*"`` to match a specific revision,
                     or a specific revision like ``"1234"``.
        :param only_recipe: If ``True``, only the recipe part of the expression is parsed,
                            ignoring ``package_id`` and ``prev``. This is useful for commands that
                            only operate on recipes, like ``conan search``.
        """
        def split(s, c, default=None):
            if not s:
                return None, default
            tokens = s.split(c, 1)
            if len(tokens) == 2:
                return tokens[0], tokens[1] or default
            return tokens[0], default

        recipe, package = split(expression, ":")
        self.raw = expression
        self.ref, rrev = split(recipe, "#", rrev)
        ref, user_channel = split(self.ref, "@")
        self.name, self.version = split(ref, "/")
        self.user, self.channel = split(user_channel, "/")
        self.rrev, _ = split(rrev, "%")
        self.package_id, prev = split(package, "#", prev)
        self.prev, _ = split(prev, "%")
        if only_recipe:
            if self.package_id:
                raise ConanException("Do not specify 'package_id' with 'only-recipe'")
        else:
            self.package_id = self.package_id or package_id

    @staticmethod
    def _only_latest(rev):
        return rev in ["!latest", "~latest"]

    @property
    def search_ref(self):
        vrange = self._version_range
        if vrange:
            return str(RecipeReference(self.name, "*", self.user, self.channel))
        if "*" in self.ref or not self.version or (self.package_id is None and self.rrev is None):
            return self.ref

    @property
    def _version_range(self):
        if self.version and self.version.startswith("[") and self.version.endswith("]"):
            return VersionRange(self.version[1:-1])

    def filter_versions(self, refs, resolve_prereleases=None):
        vrange = self._version_range
        if vrange:
            refs = [r for r in refs if vrange.contains(r.version, resolve_prereleases)]
        return refs

    @property
    def is_latest_rrev(self):
        return self.rrev == "latest"

    @property
    def is_latest_prev(self):
        return self.prev == "latest"

    def check_refs(self, refs):
        if not refs and self.ref and "*" not in self.ref:
            raise NotFoundException(f"Recipe '{self.ref}' not found")

    def filter_rrevs(self, rrevs):
        if self._only_latest(self.rrev):
            return rrevs[1:]
        rrevs = [r for r in rrevs if fnmatch.fnmatch(r.revision, self.rrev)]
        if not rrevs:
            refs_str = f'{self.ref}#{self.rrev}'
            if "*" not in refs_str:
                raise ConanException(f"Recipe revision '{refs_str}' not found")
        return rrevs

    def filter_prefs(self, prefs):
        prefs = [p for p in prefs if fnmatch.fnmatch(p.package_id, self.package_id)]
        if not prefs:
            refs_str = f'{self.ref}#{self.rrev}:{self.package_id}'
            if "*" not in refs_str:
                raise ConanException(f"Package ID '{self.raw}' not found")
        return prefs

    def filter_prevs(self, prevs):
        if self._only_latest(self.prev):
            return prevs[1:]
        prevs = [p for p in prevs if fnmatch.fnmatch(p.revision, self.prev)]
        if not prevs:
            refs_str = f'{self.ref}#{self.rrev}:{self.package_id}#{self.prev}'
            if "*" not in refs_str:
                raise ConanException(f"Package revision '{self.raw}' not found")
        return prevs
