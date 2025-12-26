import os
from collections import OrderedDict
from typing import Dict

from conan.api.model import PackagesList, MultiPackagesList, ListPattern, Remote
from conan.api.output import ConanOutput, TimedOutput
from conan.internal.api.list.query_parse import filter_package_configs
from conan.internal.conan_app import ConanBasicApp
from conan.internal.model.recipe_ref import ref_matches
from conan.internal.paths import CONANINFO
from conan.internal.errors import NotFoundException
from conan.errors import ConanException
from conan.internal.model.info import load_binary_info
from conan.api.model import PkgReference
from conan.api.model import RecipeReference
from conan.internal.util.dates import timestamp_now
from conan.internal.util.files import load


def _timelimit(expression):
    """ convert an expression like "2d" (2 days) or "3h" (3 hours) to a timestamp in the past
    with respect to current time
    """
    time_value = expression[:-1]
    try:
        time_value = int(time_value)
    except TypeError:
        raise ConanException(f"Time value '{time_value}' must be an integer")
    time_units = expression[-1]
    units = {"y": 365 * 24 * 60 * 60,
             "M": 30 * 24 * 60 * 60,
             "w": 7 * 24 * 60 * 60,
             "d": 24 * 60 * 60,
             "h": 60 * 60,
             "m": 60,
             "s": 1}
    try:
        lru_value = time_value * units[time_units]
    except KeyError:
        raise ConanException(f"Unrecognized time unit: '{time_units}'. Use: {list(units)}")

    limit = timestamp_now() - lru_value
    return limit


class ListAPI:
    """ Get references from the recipes and packages in the cache or a remote
    """

    def __init__(self, conan_api):
        self._conan_api = conan_api

    def latest_recipe_revision(self, ref: RecipeReference, remote: Remote = None):
        """ For a given recipe reference, return the latest revision of the recipe in the remote,
        or in the local cache if no remote is specified, or ``None`` if the recipe does not exist."""
        assert ref.revision is None, "latest_recipe_revision: ref already have a revision"
        app = ConanBasicApp(self._conan_api)
        if remote:
            ret = app.remote_manager.get_latest_recipe_revision(ref, remote=remote)
        else:
            ret = app.cache.get_latest_recipe_revision(ref)

        return ret

    def recipe_revisions(self, ref: RecipeReference, remote: Remote = None):
        """ For a given recipe reference, return all the revisions of the recipe in the remote,
        or in the local cache if no remote is specified"""
        assert ref.revision is None, "recipe_revisions: ref already have a revision"
        app = ConanBasicApp(self._conan_api)
        if remote:
            results = app.remote_manager.get_recipe_revisions(ref, remote=remote)
        else:
            results = app.cache.get_recipe_revisions(ref)

        return results

    def latest_package_revision(self, pref: PkgReference, remote=None):
        # TODO: This returns None if the given package_id is not existing. It should probably
        #  raise NotFound, but to keep aligned with the above ``latest_recipe_revision`` which
        #  is used as an "exists" check too in other places, lets respect the None return
        assert pref.revision is None, "latest_package_revision: ref already have a revision"
        assert pref.package_id is not None, "package_id must be defined"
        app = ConanBasicApp(self._conan_api)
        if remote:
            ret = app.remote_manager.get_latest_package_revision(pref, remote=remote)
        else:
            ret = app.cache.get_latest_package_revision(pref)
        return ret

    def package_revisions(self, pref: PkgReference, remote=None):
        assert pref.ref.revision is not None, "package_revisions requires a recipe revision, " \
                                              "check latest first if needed"
        app = ConanBasicApp(self._conan_api)
        if remote:
            results = app.remote_manager.get_package_revisions(pref, remote=remote)
        else:
            results = app.cache.get_package_revisions(pref)
        return results

    def _packages_configurations(self, ref: RecipeReference,
                                 remote=None) -> Dict[PkgReference, dict]:
        assert ref.revision is not None and ref.revision != "latest", \
            "packages: ref should have a revision. Check latest if needed."
        app = ConanBasicApp(self._conan_api)
        if not remote:
            prefs = app.cache.get_package_references(ref)
            packages = _get_cache_packages_binary_info(app.cache, prefs)
        else:
            packages = app.remote_manager.search_packages(remote, ref)
        return packages

    @staticmethod
    def _filter_packages_configurations(pkg_configurations, query):
        """
        :param pkg_configurations: Dict[PkgReference, PkgConfiguration]
        :param query: str like "os=Windows AND (arch=x86 OR compiler=gcc)"
        :return: Dict[PkgReference, PkgConfiguration]
        """
        try:
            if "!" in query:
                raise ConanException("'!' character is not allowed")
            if "~" in query:
                raise ConanException("'~' character is not allowed")
            if " not " in query or query.startswith("not "):
                raise ConanException("'not' operator is not allowed")
            return filter_package_configs(pkg_configurations, query)
        except Exception as exc:
            raise ConanException("Invalid package query: %s. %s" % (query, exc))

    @staticmethod
    def _filter_packages_profile(packages, profile, ref):
        result = {}
        profile_settings = profile.processed_settings.serialize()
        # Options are those for dependencies, like *:shared=True
        profile_options = profile.options._deps_package_options
        for pref, data in packages.items():
            settings = data.get("settings", {})
            settings_match = options_match = True
            for k, v in settings.items():  # Only the defined settings that don't match
                value = profile_settings.get(k)
                if value is not None and value != v:
                    settings_match = False
                    break
            options = data.get("options", {})
            for k, v in options.items():
                for pattern, pattern_options in profile_options.items():
                    # Accept &: as referring to the current package being listed,
                    # even if it's not technically a "consumer"
                    if ref_matches(ref, pattern, True):
                        value = pattern_options.get_safe(k)
                        if value is not None and value != v:
                            options_match = False
                            break

            if settings_match and options_match:
                result[pref] = data

        return result

    def select(self, pattern: ListPattern, package_query=None, remote: Remote = None, lru=None,
               profile=None) -> PackagesList:
        """For a given pattern, return a list of recipes and packages matching the provided filters.

        :parameter ListPattern pattern: Search criteria
        :parameter str package_query: When returning packages, expression of the form
            ``"os=Windows AND (arch=x86 OR compiler=gcc)"`` to filter packages by.
            If ``None``, all packages will be returned if requested.
        :parameter Remote remote: Remote to search in,
            if ``None``, it will search in the local cache.
        :parameter str lru: If set, it will filter the results to only include
            packages/binaries that have been used in the last 'lru' time.
            It can be a string like ``"2d"`` (2 days) or ``"3h"`` (3 hours).
        :parameter Profile profile: Profile to filter the packages by settings and options.
        """
        # TODO: Implement better error forwarding for "list" command that captures Exceptions
        if package_query and pattern.package_id and "*" not in pattern.package_id:
            raise ConanException("Cannot specify '-p' package queries, "
                                 "if 'package_id' is not a pattern")
        if remote and lru:
            raise ConanException("'--lru' cannot be used in remotes, only in cache")

        select_bundle = PackagesList()
        # Avoid doing a ``search`` of recipes if it is an exact ref and it will be used later
        search_ref = pattern.search_ref
        app = ConanBasicApp(self._conan_api)
        limit_time = _timelimit(lru) if lru else None
        out = ConanOutput()
        remote_name = "local cache" if not remote else remote.name
        if search_ref:
            refs = _search_recipes(app, search_ref, remote=remote)
            global_conf = self._conan_api._api_helpers.global_conf  # noqa
            resolve_prereleases = global_conf.get("core.version_ranges:resolve_prereleases")
            refs = pattern.filter_versions(refs, resolve_prereleases)
            pattern.check_refs(refs)
            out.info(f"Found {len(refs)} pkg/version recipes matching {search_ref} in {remote_name}")
        else:
            refs = [RecipeReference(pattern.name, pattern.version, pattern.user, pattern.channel)]

        # Show only the recipe references
        if pattern.package_id is None and pattern.rrev is None:
            for r in refs:
                select_bundle.add_ref(r)
            return select_bundle

        def msg_format(msg, item, total):
            return msg + f" ({total.index(item)}/{len(total)})"

        trefs = TimedOutput(5, msg_format=msg_format)
        for r in refs:  # Older versions first
            trefs.info(f"Listing revisions of {r} in {remote_name}", r, refs)
            if pattern.is_latest_rrev or pattern.rrev is None:
                rrev = self.latest_recipe_revision(r, remote)
                if rrev is None:
                    raise NotFoundException(f"Recipe '{r}' not found")
                rrevs = [rrev]
            else:
                rrevs = self.recipe_revisions(r, remote)
                rrevs = pattern.filter_rrevs(rrevs)
                rrevs = list(reversed(rrevs))  # Order older revisions first

            if lru and pattern.package_id is None:  # Filter LRUs
                rrevs = [r for r in rrevs if app.cache.get_recipe_lru(r) < limit_time]

            for rr in rrevs:
                select_bundle.add_ref(rr)

            if pattern.package_id is None:  # Stop if not displaying binaries
                continue

            trrevs = TimedOutput(5, msg_format=msg_format)
            for rrev in rrevs:
                trrevs.info(f"Listing binaries of {rrev.repr_notime()} in {remote_name}", rrev, rrevs)
                prefs = []
                if "*" not in pattern.package_id and pattern.prev is not None:
                    prefs.append(PkgReference(rrev, package_id=pattern.package_id))
                    packages = {}
                else:
                    packages = self._packages_configurations(rrev, remote)
                    if package_query is not None:
                        packages = self._filter_packages_configurations(packages, package_query)
                    if profile is not None:
                        packages = self._filter_packages_profile(packages, profile, rrev)
                    prefs = packages.keys()
                    prefs = pattern.filter_prefs(prefs)
                    packages = {pref: conf for pref, conf in packages.items() if pref in prefs}

                if pattern.prev is not None:
                    new_prefs = []
                    for pref in prefs:
                        # Maybe the package_configurations returned timestamp
                        if pattern.is_latest_prev or pattern.prev is None:
                            prev = self.latest_package_revision(pref, remote)
                            if prev is None:
                                raise NotFoundException(f"Binary package not found: '{pref}")
                            new_prefs.append(prev)
                        else:
                            prevs = self.package_revisions(pref, remote)
                            prevs = pattern.filter_prevs(prevs)
                            prevs = list(reversed(prevs))  # Older revisions first
                            new_prefs.extend(prevs)
                    prefs = new_prefs

                if lru:  # Filter LRUs
                    prefs = [r for r in prefs if app.cache.get_package_lru(r) < limit_time]

                # Packages dict has been listed, even if empty
                select_bundle.recipe_dict(rrev)["packages"] = {}
                for p in prefs:
                    # the "packages" dict is not using the package-revision
                    pkg_info = packages.get(PkgReference(p.ref, p.package_id))
                    select_bundle.add_pref(p, pkg_info)
        return select_bundle

    def explain_missing_binaries(self, ref, conaninfo, remotes):
        """ (Experimental) Explain why a binary is missing in the cache
        """
        ConanOutput().info(f"Missing binary: {ref}")
        ConanOutput().info(f"With conaninfo.txt (package_id):\n{conaninfo.dumps()}")
        conaninfo = load_binary_info(conaninfo.dumps())
        # Collect all configurations
        candidates = []
        ConanOutput().info(f"Finding binaries in the cache")
        pkg_configurations = self._packages_configurations(ref)
        candidates.extend(_BinaryDistance(pref, data, conaninfo)
                          for pref, data in pkg_configurations.items())

        for remote in remotes:
            try:
                ConanOutput().info(f"Finding binaries in remote {remote.name}")
                pkg_configurations = self._packages_configurations(ref, remote=remote)
            except Exception as e:
                ConanOutput().error(f"Error in remote '{remote.name}': {e}")
            else:
                candidates.extend(_BinaryDistance(pref, data, conaninfo, remote)
                                  for pref, data in pkg_configurations.items())

        candidates.sort()
        pkglist = PackagesList()
        pkglist.add_ref(ref)
        # Return the closest matches, stop adding when distance is increased
        candidate_distance = None
        for candidate in candidates:
            if candidate_distance and candidate.distance != candidate_distance:
                break
            candidate_distance = candidate.distance
            pref = candidate.pref
            pkglist.add_pref(pref, candidate.binary_config)
            # Add the diff data
            rev_dict = pkglist.recipe_dict(ref)
            rev_dict["packages"][pref.package_id]["diff"] = candidate.serialize()
            remote = candidate.remote.name if candidate.remote else "Local Cache"
            rev_dict["packages"][pref.package_id]["remote"] = remote
        return pkglist

    def find_remotes(self, package_list, remotes):
        """
        (Experimental) Find the remotes where the current package lists can be found
        """
        result = MultiPackagesList()
        app = ConanBasicApp(self._conan_api)
        for r in remotes:
            result_pkg_list = PackagesList()
            for ref, ref_contents in package_list.serialize().items():
                ref = RecipeReference.loads(ref)
                try:
                    remote_rrevs = app.remote_manager.get_recipe_revisions(ref, remote=r)
                except NotFoundException:
                    continue
                revisions = ref_contents.get("revisions")
                if revisions is None:  # This is a package list just with name/version
                    if remote_rrevs:
                        result_pkg_list.add_ref(ref)
                    continue

                for revision, rev_content in revisions.items():
                    ref.revision = revision
                    # We look for the value of revision in server, to return timestamp too
                    found = next((r for r in remote_rrevs if r == ref), None)
                    if not found:
                        continue
                    result_pkg_list.add_ref(found)
                    packages = rev_content.get("packages")
                    if packages is None:
                        continue
                    for pkgid, pkgcontent in packages.items():
                        pref = PkgReference(ref, pkgid)
                        try:
                            remote_prefs = app.remote_manager.get_package_revisions(pref, remote=r)
                        except NotFoundException:
                            continue
                        pkg_revisions = pkgcontent.get("revisions")
                        if pkg_revisions is None:  # This is a package_id, no prevs
                            if remote_prefs:
                                result_pkg_list.add_pref(pref, pkgcontent.get("info"))
                            continue
                        for pkg_revision, prev_content in pkg_revisions.items():
                            pref.revision = pkg_revision
                            # We look for the value of revision in server, to return timestamp too
                            pfound = next((r for r in remote_prefs if r == pref), None)
                            if not pfound:
                                continue
                            result_pkg_list.add_pref(pfound, pkgcontent.get("info"))

            if result_pkg_list:
                result.add(r.name, result_pkg_list)
        return result

    def outdated(self, deps_graph, remotes):
        # DO NOT USE YET
        # Data structure to store info per library
        dependencies = deps_graph.nodes[1:]
        dict_nodes = {}

        # When there are no dependencies command should stop
        if len(dependencies) == 0:
            return dict_nodes

        ConanOutput().title("Checking remotes")

        for node in dependencies:
            dict_nodes.setdefault(node.name, {"cache_refs": set(), "version_ranges": [],
                                              "latest_remote": None})["cache_refs"].add(node.ref)

        for version_range in deps_graph.resolved_ranges.keys():
            dict_nodes[version_range.name]["version_ranges"].append(version_range)

        # find in remotes
        for node_name, node_info in dict_nodes.items():
            ref_pattern = ListPattern(node_name, rrev=None, prev=None)
            for remote in remotes:
                try:
                    remote_ref_list = self.select(ref_pattern, package_query=None, remote=remote)
                except NotFoundException:
                    continue
                if not remote_ref_list:
                    continue
                str_latest_ref = list(remote_ref_list.serialize().keys())[-1]
                recipe_ref = RecipeReference.loads(str_latest_ref)
                if (node_info["latest_remote"] is None
                        or node_info["latest_remote"]["ref"] < recipe_ref):
                    node_info["latest_remote"] = {"ref": recipe_ref, "remote": remote.name}

        # Filter nodes with no outdated versions
        filtered_nodes = {}
        for node_name, node in dict_nodes.items():
            if node['latest_remote'] is not None and sorted(list(node['cache_refs']))[0] < \
                    node['latest_remote']['ref']:
                filtered_nodes[node_name] = node

        return filtered_nodes


class _BinaryDistance:
    def __init__(self, pref, binary, expected, remote=None):
        self.remote = remote
        self.pref = pref
        self.binary_config = binary

        # Settings, special handling for os/arch
        binary_settings = binary.get("settings", {})
        expected_settings = expected.get("settings", {})

        platform = {k: v for k, v in binary_settings.items() if k in ("os", "arch")}
        expected_platform = {k: v for k, v in expected_settings.items() if k in ("os", "arch")}
        self.platform_diff = self._calculate_diff(platform, expected_platform)

        binary_settings = {k: v for k, v in binary_settings.items() if k not in ("os", "arch")}
        expected_settings = {k: v for k, v in expected_settings.items() if k not in ("os", "arch")}
        self.settings_diff = self._calculate_diff(binary_settings, expected_settings)

        self.settings_target_diff = self._calculate_diff(binary, expected, "settings_target")
        self.options_diff = self._calculate_diff(binary, expected, "options")
        self.deps_diff = self._requirement_diff(binary, expected, "requires")
        self.build_requires_diff = self._requirement_diff(binary, expected, "build_requires")
        self.python_requires_diff = self._requirement_diff(binary, expected, "python_requires")
        self.confs_diff = self._calculate_diff(binary,  expected, "conf")

    @staticmethod
    def _requirement_diff(binary_requires, expected_requires, item):
        binary_requires = binary_requires.get(item, {})
        expected_requires = expected_requires.get(item, {})
        output = {}
        binary_requires = [RecipeReference.loads(r) for r in binary_requires]
        expected_requires = [RecipeReference.loads(r) for r in expected_requires]
        binary_requires = {r.name: r for r in binary_requires}
        for r in expected_requires:
            existing = binary_requires.get(r.name)
            if not existing or r != existing:
                output.setdefault("expected", []).append(repr(r))
                output.setdefault("existing", []).append(repr(existing))
        expected_requires = {r.name: r for r in expected_requires}
        for r in binary_requires.values():
            existing = expected_requires.get(r.name)
            if not existing or r != existing:
                if repr(existing) not in output.get("expected", ()):
                    output.setdefault("expected", []).append(repr(existing))
                if repr(r) not in output.get("existing", ()):
                    output.setdefault("existing", []).append(repr(r))
        return output

    @staticmethod
    def _calculate_diff(binary_confs, expected_confs, item=None):
        if item is not None:
            binary_confs = binary_confs.get(item, {})
            expected_confs = expected_confs.get(item, {})
        output = {}
        for k, v in expected_confs.items():
            value = binary_confs.get(k)
            if value != v:
                output.setdefault("expected", []).append(f"{k}={v}")
                output.setdefault("existing", []).append(f"{k}={value}")
        for k, v in binary_confs.items():
            value = expected_confs.get(k)
            if value != v:
                if f"{k}={value}" not in output.get("expected", ()):
                    output.setdefault("expected", []).append(f"{k}={value}")
                if f"{k}={v}" not in output.get("existing", ()):
                    output.setdefault("existing", []).append(f"{k}={v}")
        return output

    def __lt__(self, other):
        return self.distance < other.distance

    def explanation(self):
        if self.platform_diff:
            return "This binary belongs to another OS or Architecture, highly incompatible."
        if self.settings_diff:
            return "This binary was built with different settings."
        if self.settings_target_diff:
            return "This binary was built with different settings_target."
        if self.options_diff:
            return "This binary was built with the same settings, but different options"
        if self.deps_diff:
            return "This binary has same settings and options, but different dependencies"
        if self.build_requires_diff:
            return "This binary has same settings, options and dependencies, but different build_requires"
        if self.python_requires_diff:
            return "This binary has same settings, options and dependencies, but different python_requires"
        if self.confs_diff:
            return "This binary has same settings, options and dependencies, but different confs"
        return "This binary is an exact match for the defined inputs"

    @property
    def distance(self):
        return (len(self.platform_diff.get("expected", [])),
                len(self.settings_diff.get("expected", [])),
                len(self.settings_target_diff.get("expected", [])),
                len(self.options_diff.get("expected", [])),
                len(self.deps_diff.get("expected", [])),
                len(self.build_requires_diff.get("expected", [])),
                len(self.python_requires_diff.get("expected", [])),
                len(self.confs_diff.get("expected", [])))

    def serialize(self):
        return {"platform": self.platform_diff,
                "settings": self.settings_diff,
                "settings_target": self.settings_target_diff,
                "options": self.options_diff,
                "dependencies": self.deps_diff,
                "build_requires": self.build_requires_diff,
                "python_requires": self.python_requires_diff,
                "confs": self.confs_diff,
                "explanation": self.explanation()}


def _get_cache_packages_binary_info(cache, prefs) -> Dict[PkgReference, dict]:
    """
    param package_layout: Layout for the given reference
    """

    result = OrderedDict()

    for pref in prefs:
        latest_prev = cache.get_latest_package_revision(pref)
        pkg_layout = cache.pkg_layout(latest_prev)

        # Read conaninfo
        info_path = os.path.join(pkg_layout.package(), CONANINFO)
        if not os.path.exists(info_path):
            ConanOutput().error(f"Corrupted package '{pkg_layout.reference}' "
                                f"without conaninfo.txt in: {info_path}")
            info = {}
        else:
            conan_info_content = load(info_path)
            info = load_binary_info(conan_info_content)
        pref = pkg_layout.reference
        # The key shoudln't have the latest package revision, we are asking for package configs
        pref.revision = None
        result[pkg_layout.reference] = info

    return result


def _search_recipes(app, query: str, remote=None):
    only_none_user_channel = False
    if query and query.endswith("@"):
        only_none_user_channel = True
        query = query[:-1]

    if remote:
        refs = app.remote_manager.search_recipes(remote, query)
    else:
        refs = app.cache.search_recipes(query)
    ret = []
    for r in refs:
        if not only_none_user_channel or (r.user is None and r.channel is None):
            ret.append(r)
    return sorted(ret)
