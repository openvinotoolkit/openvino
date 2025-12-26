from conan.api.output import ConanOutput
from conan.internal.cache.conan_reference_layout import BasicLayout
from conan.internal.graph.graph import (RECIPE_DOWNLOADED, RECIPE_INCACHE, RECIPE_NEWER,
                                        RECIPE_NOT_IN_REMOTE, RECIPE_UPDATED, RECIPE_EDITABLE,
                                        RECIPE_INCACHE_DATE_UPDATED, RECIPE_UPDATEABLE)
from conan.internal.errors import NotFoundException
from conan.errors import ConanException


class ConanProxy:
    def __init__(self, conan_app, editable_packages, legacy_update=None):
        # collaborators
        self._editable_packages = editable_packages
        self._cache = conan_app.cache
        self._remote_manager = conan_app.remote_manager
        self._resolved = {}  # Cache of the requested recipes to optimize calls
        self._legacy_update = legacy_update

    def get_recipe(self, ref, remotes, update, check_update):
        """
        :return: Tuple (layout, status, remote)
        """
        # TODO: cache2.0 Check with new locks
        # with layout.conanfile_write_lock(self._out):
        resolved = self._resolved.get(ref)
        if resolved is None:
            resolved = self._get_recipe(ref, remotes, update, check_update)
            self._resolved[ref] = resolved
        return resolved

    # return the remote where the recipe was found or None if the recipe was not found
    def _get_recipe(self, reference, remotes, update, check_update):
        output = ConanOutput(scope=str(reference))

        conanfile_path = self._editable_packages.get_path(reference)
        if conanfile_path is not None:
            return BasicLayout(reference, conanfile_path), RECIPE_EDITABLE, None

        # check if it there's any revision of this recipe in the local cache
        try:
            cache_ref = self._cache.get_latest_recipe_revision(reference) \
                if reference.revision is None else reference
            recipe_layout = self._cache.recipe_layout(cache_ref)
            ref = recipe_layout.reference  # latest revision if it was not defined
        except ConanException:
            # NOT in disk, must be retrieved from remotes
            # we will only check all servers for latest revision if we did a --update
            layout, remote = self._download_recipe(reference, remotes, output, update, check_update)
            status = RECIPE_DOWNLOADED
            return layout, status, remote

        self._cache.update_recipe_lru(ref)

        # TODO: cache2.0: check with new --update flows
        # TODO: If the revision is given, then we don't need to check for updates?
        if not (check_update or should_update_reference(reference, update)):
            status = RECIPE_INCACHE
            return recipe_layout, status, None

        # Need to check updates
        remote, remote_ref = self._find_newest_recipe_in_remotes(reference, remotes,
                                                                 update, check_update)
        if remote_ref is None:  # Nothing found in remotes
            status = RECIPE_NOT_IN_REMOTE
            return recipe_layout, status, None

        # Something found in remotes, check if we already have the latest in local cache
        # TODO: cache2.0 here if we already have a revision in the cache but we add the
        #  --update argument and we find that same revision in server, we will not
        #  download anything but we will UPDATE the date of that revision in the
        #  local cache and WE ARE ALSO UPDATING THE REMOTE
        #  Check if this is the flow we want to follow
        assert ref.timestamp
        cache_time = ref.timestamp
        if remote_ref.revision != ref.revision:
            if cache_time < remote_ref.timestamp:
                # the remote one is newer
                if should_update_reference(remote_ref, update):
                    output.info("Retrieving from remote '%s'..." % remote.name)
                    new_recipe_layout = self._download(remote_ref, remote)
                    status = RECIPE_UPDATED
                    return new_recipe_layout, status, remote
                else:
                    status = RECIPE_UPDATEABLE
            else:
                status = RECIPE_NEWER
                # If your recipe in cache is newer it does not make sense to return a remote?
                remote = None
        else:
            # TODO: cache2.0 we are returning RECIPE_UPDATED just because we are updating
            #  the date
            if cache_time >= remote_ref.timestamp:
                status = RECIPE_INCACHE
            else:
                self._cache.update_recipe_timestamp(remote_ref)
                status = RECIPE_INCACHE_DATE_UPDATED
        return recipe_layout, status, remote

    def _find_newest_recipe_in_remotes(self, reference, remotes, update, check_update):
        output = ConanOutput(scope=str(reference))

        results = []
        need_update = should_update_reference(reference, update) or check_update
        for remote in remotes:
            if remote.allowed_packages and not any(reference.matches(f, is_consumer=False)
                                                   for f in remote.allowed_packages):
                output.debug(f"Excluding remote {remote.name} because recipe is filtered out")
                continue
            output.info(f"Checking remote: {remote.name}")
            try:
                if self._legacy_update and need_update and reference.revision is None:
                    if not getattr(ConanProxy, "update_policy_legacy_warning", None):
                        ConanProxy.update_policy_legacy_warning = True
                        ConanOutput().warning("The 'core:update_policy' conf is deprecated and will "
                                              "be removed in future versions", warn_tag="deprecated")
                    refs = self._remote_manager.get_recipe_revisions(reference, remote)
                    results.extend([{'remote': remote, 'ref': ref} for ref in refs])
                    continue
                if not reference.revision:
                    ref = self._remote_manager.get_latest_recipe_revision(reference, remote)
                else:
                    ref = self._remote_manager.get_recipe_revision(reference, remote)
                if not need_update:
                    return remote, ref
                results.append({'remote': remote, 'ref': ref})
            except NotFoundException:
                pass

        if len(results) == 0:
            return None, None

        if self._legacy_update and need_update:
            # Use only the first occurence of each revision in the remotes
            filtered_results = []
            revisions = set()
            for r in results:
                ref = r["ref"]
                if ref.revision not in revisions:
                    revisions.add(ref.revision)
                    filtered_results.append(r)
            results = filtered_results

        remotes_results = sorted(results, key=lambda k: k['ref'].timestamp, reverse=True)
        # get the latest revision from all remotes
        found_rrev = remotes_results[0]
        return found_rrev.get("remote"), found_rrev.get("ref")

    def _download_recipe(self, ref, remotes, scoped_output, update, check_update):
        # When a recipe doesn't existin local cache, it is retrieved from servers
        scoped_output.info("Not found in local cache, looking in remotes...")
        if not remotes:
            raise ConanException("No remote defined")

        remote, latest_rref = self._find_newest_recipe_in_remotes(ref, remotes, update, check_update)
        if not latest_rref:
            msg = "Unable to find '%s' in remotes" % repr(ref)
            raise NotFoundException(msg)

        recipe_layout = self._download(latest_rref, remote)
        return recipe_layout, remote

    def _download(self, ref, remote):
        assert ref.revision
        assert ref.timestamp
        recipe_layout = self._remote_manager.get_recipe(ref, remote)
        output = ConanOutput(scope=str(ref))
        output.info("Downloaded recipe revision %s" % ref.revision)
        return recipe_layout


def should_update_reference(reference, update):
    if update is None:
        return False
    # Old API usages only ever passed a bool
    if isinstance(update, bool):
        return update
    # Legacy syntax had --update without pattern, it manifests as a "*" pattern
    return any(name == "*" or reference.name == name for name in update)
