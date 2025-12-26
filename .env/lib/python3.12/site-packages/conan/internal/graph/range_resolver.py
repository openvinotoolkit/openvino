from conan.api.output import ConanOutput
from conan.internal.graph.proxy import should_update_reference
from conan.errors import ConanException
from conan.api.model import RecipeReference
from conan.internal.model.version_range import VersionRange


class RangeResolver:

    def __init__(self, conan_app, global_conf, editable_packages):
        self._cache = conan_app.cache
        self._editable_packages = editable_packages
        self._remote_manager = conan_app.remote_manager
        self._cached_cache = {}  # Cache caching of search result, so invariant wrt installations
        self._cached_remote_found = {}  # dict {ref (pkg/*): {remote_name: results (pkg/1, pkg/2)}}
        self.resolved_ranges = {}
        self._resolve_prereleases = global_conf.get('core.version_ranges:resolve_prereleases')

    def resolve(self, require, base_conanref, remotes, update):
        try:
            version_range = require.version_range
        except Exception as e:
            base = base_conanref or "conanfile"
            raise ConanException(f"\n    Recipe '{base}' requires '{require.ref}' "
                                 f"version-range definition error:\n    {e}")
        if version_range is None:
            return
        assert isinstance(version_range, VersionRange)

        if require.ref.revision is not None:
            ConanOutput(base_conanref).warning("Specifying a revision for requirement "
                                               f"'{require.ref.repr_notime()}' together with "
                                               "a version range has no effect. "
                                               "The revision will be ignored.",
                                               warn_tag="risk")

        # Check if this ref with version range was already solved
        previous_ref = self.resolved_ranges.get(require.ref)
        if previous_ref is not None:
            require.ref = previous_ref
            return

        ref = require.ref
        search_ref = RecipeReference(ref.name, "*", ref.user, ref.channel)

        resolved_ref = self._resolve_local(search_ref, version_range)
        if resolved_ref is None or should_update_reference(search_ref, update):
            remote_resolved_ref = self._resolve_remote(search_ref, version_range, remotes, update)
            if resolved_ref is None or (remote_resolved_ref is not None and
                                        resolved_ref.version < remote_resolved_ref.version):
                resolved_ref = remote_resolved_ref

        if resolved_ref is None:
            raise ConanException(f"Version range '{version_range}' from requirement '{require.ref}' "
                                 f"required by '{base_conanref}' could not be resolved")

        # To fix Cache behavior, we remove the revision information
        resolved_ref.revision = None  # FIXME: Wasting information already obtained from server?
        self.resolved_ranges[require.ref] = resolved_ref
        require.ref = resolved_ref

    def _resolve_local(self, search_ref, version_range):
        pattern = str(search_ref)
        local_found = self._cached_cache.get(pattern)
        if local_found is None:
            # This local_found is weird, it contains multiple revisions, not just latest
            local_found = self._cache.search_recipes(pattern)
            # TODO: This is still necessary to filter user/channel, until search_recipes is fixed
            local_found = [ref for ref in local_found if ref.user == search_ref.user
                           and ref.channel == search_ref.channel]
            local_found.extend(r for r in self._editable_packages.edited_refs
                               if r.name == search_ref.name and r.user == search_ref.user
                               and r.channel == search_ref.channel)
            self._cached_cache[pattern] = local_found
        if local_found:
            return self._resolve_version(version_range, local_found, self._resolve_prereleases)

    def _search_remote_recipes(self, remote, search_ref):
        if remote.allowed_packages and not any(search_ref.matches(f, is_consumer=False)
                                               for f in remote.allowed_packages):
            return []
        pattern = str(search_ref)
        pattern_cached = self._cached_remote_found.setdefault(pattern, {})
        results = pattern_cached.get(remote.name)
        if results is None:
            results = self._remote_manager.search_recipes(remote, pattern)
            # TODO: This is still necessary to filter user/channel, until search_recipes is fixed
            results = [ref for ref in results if ref.user == search_ref.user
                       and ref.channel == search_ref.channel]
            pattern_cached.update({remote.name: results})
        return results

    def _resolve_remote(self, search_ref, version_range, remotes, update):
        update_candidates = []
        for remote in remotes:
            remote_results = self._search_remote_recipes(remote, search_ref)
            resolved_version = self._resolve_version(version_range, remote_results,
                                                     self._resolve_prereleases)
            if resolved_version:
                if not should_update_reference(search_ref, update):
                    return resolved_version  # Return first valid occurrence in first remote
                else:
                    update_candidates.append(resolved_version)
        if len(update_candidates) > 0:  # pick latest from already resolved candidates
            resolved_version = self._resolve_version(version_range, update_candidates,
                                                     self._resolve_prereleases)
            return resolved_version

    @staticmethod
    def _resolve_version(version_range, refs_found, resolve_prereleases):
        for ref in reversed(sorted(refs_found)):
            if version_range.contains(ref.version, resolve_prereleases):
                return ref
