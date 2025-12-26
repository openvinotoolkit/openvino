from urllib.parse import urlencode

from conan.api.model import RecipeReference
from conan.internal.rest.rest_routes import RestRoutes


def _format_ref(url, ref):
    url = url.format(name=ref.name, version=ref.version, username=ref.user or "_",
                     channel=ref.channel or "_", revision=ref.revision)
    return url


def _format_pref(url, pref):
    ref = pref.ref
    url = url.format(name=ref.name, version=ref.version, username=ref.user or "_",
                     channel=ref.channel or "_", revision=ref.revision, package_id=pref.package_id,
                     p_revision=pref.revision)
    return url


class ClientV2Router:
    """Builds urls for v2"""

    def __init__(self, root_url):
        self.root_url = root_url
        self.base_url = "{}/v2/".format(root_url)
        self.routes = RestRoutes()

    def ping(self):
        # FIXME: The v2 ping is not returning capabilities
        return "{}/v1/".format(self.root_url) + self.routes.ping

    def search(self, pattern, ignorecase):
        """URL search recipes"""
        query = ''
        if pattern:
            if isinstance(pattern, RecipeReference):
                pattern = repr(pattern)
            params = {"q": pattern}
            if not ignorecase:
                params["ignorecase"] = "False"
            query = "?%s" % urlencode(params)
        return self.base_url + "%s%s" % (self.routes.common_search, query)

    def search_packages(self, ref, list_only):
        """URL search packages for a recipe"""
        route = self.routes.common_search_packages_revision \
            if ref.revision else self.routes.common_search_packages
        url = _format_ref(route, ref)

        params = {"list_only": list_only}
        query = "?%s" % urlencode(params)
        return self.base_url + url + query

    def common_authenticate(self):
        return self.base_url + self.routes.common_authenticate

    def common_check_credentials(self):
        return self.base_url + self.routes.common_check_credentials

    def recipe_file(self, ref, path):
        """Recipe file url"""
        return self.base_url + self._for_recipe_file(ref, path)

    def package_file(self, pref, path):
        """Package file url"""
        return self.base_url + self._for_package_file(pref, path)

    def remove_recipe(self, ref):
        """Remove recipe url"""
        return self.base_url + self._for_recipe(ref)

    def recipe_revisions(self, ref):
        """Get revisions for a recipe url"""
        return self.base_url + _format_ref(self.routes.recipe_revisions, ref)

    def remove_package(self, pref):
        """Remove package url"""
        assert pref.revision is not None, "remove_package v2 needs PREV"
        return self.base_url + self._for_package(pref)

    def remove_all_packages(self, ref):
        """Remove package url"""
        return self.base_url + self._for_packages(ref)

    def recipe_snapshot(self, ref):
        """get recipe manifest url"""
        return self.base_url + self._for_recipe_files(ref)

    def package_snapshot(self, pref):
        """get recipe manifest url"""
        return self.base_url + self._for_package_files(pref)

    def package_revisions(self, pref):
        """get revisions for a package url"""
        return self.base_url + _format_pref(self.routes.package_revisions, pref)

    def package_latest(self, pref):
        """Get the latest of a package"""
        assert pref.ref.revision is not None, "Cannot get the latest package without RREV"
        return self.base_url + _format_pref(self.routes.package_revision_latest, pref)

    def recipe_latest(self, ref):
        """Get the latest of a recipe"""
        assert ref.revision is None, "for_recipe_latest shouldn't receive RREV"
        return self.base_url + _format_ref(self.routes.recipe_latest, ref)

    def _for_package_file(self, pref, path):
        """url for getting a file from a package, with revisions"""
        assert pref.ref.revision is not None, "_for_package_file needs RREV"
        assert pref.revision is not None, "_for_package_file needs PREV"
        return ClientV2Router._format_pref_path(self.routes.package_revision_file, pref, path)

    def _for_package_files(self, pref):
        """url for getting the recipe list"""
        assert pref.revision is not None, "_for_package_files needs PREV"
        assert pref.ref.revision is not None, "_for_package_files needs RREV"
        return _format_pref(self.routes.package_revision_files, pref)

    def _for_recipe_file(self, ref, path):
        """url for a recipe file, with or without revisions"""
        assert ref.revision is not None, "for_recipe_file needs RREV"
        return ClientV2Router._format_ref_path(self.routes.recipe_revision_file, ref, path)

    def _for_recipe_files(self, ref):
        """url for getting the recipe list"""
        assert ref.revision is not None, "for_recipe_files needs RREV"
        return _format_ref(self.routes.recipe_revision_files, ref)

    def _for_recipe(self, ref):
        """url for a recipe with or without revisions (without rev,
        only for delete the root recipe, or v1)"""
        return _format_ref(self.routes.recipe_revision, ref)

    def _for_packages(self, ref):
        """url for a recipe with or without revisions"""
        return _format_ref(self.routes.packages_revision, ref)

    def _for_package(self, pref):
        """url for the package with or without revisions"""
        return _format_pref(self.routes.package_revision, pref)

    @staticmethod
    def _format_ref_path(url, ref, path):
        ret = url.format(name=ref.name, version=ref.version, username=ref.user or "_",
                         channel=ref.channel or "_", revision=ref.revision, path=path)
        return ret

    @staticmethod
    def _format_pref_path(url, pref, path):
        ref = pref.ref
        return url.format(name=ref.name, version=ref.version, username=ref.user or "_",
                          channel=ref.channel or "_", revision=ref.revision,
                          package_id=pref.package_id,
                          p_revision=pref.revision, path=path)
