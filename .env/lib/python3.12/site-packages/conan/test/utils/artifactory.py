import os
import time
import uuid

import requests

from conan.internal.errors import RecipeNotFoundException, PackageNotFoundException
from conans.server.revision_list import _RevisionEntry

ARTIFACTORY_DEFAULT_USER = os.getenv("ARTIFACTORY_DEFAULT_USER", "admin")
ARTIFACTORY_DEFAULT_PASSWORD = os.getenv("ARTIFACTORY_DEFAULT_PASSWORD", "password")
ARTIFACTORY_DEFAULT_URL = os.getenv("ARTIFACTORY_DEFAULT_URL", "http://localhost:8090/artifactory")


class _ArtifactoryServerStore(object):

    def __init__(self, repo_url, user, password):
        self._user = user or ARTIFACTORY_DEFAULT_USER
        self._password = password or ARTIFACTORY_DEFAULT_PASSWORD
        self._repo_url = repo_url

    @property
    def _auth(self):
        return self._user, self._password

    @staticmethod
    def _root_recipe(ref):
        return "{}/{}/{}/{}".format(ref.user or "_", ref.name, ref.version, ref.channel or "_")

    @staticmethod
    def _ref_index(ref):
        return "{}/index.json".format(_ArtifactoryServerStore._root_recipe(ref))

    @staticmethod
    def _pref_index(pref):
        tmp = _ArtifactoryServerStore._root_recipe(pref.ref)
        return "{}/{}/package/{}/index.json".format(tmp, pref.ref.revision, pref.package_id)

    def get_recipe_revisions_references(self, ref):
        time.sleep(0.1)  # Index appears to not being updated immediately after a remove
        url = "{}/{}".format(self._repo_url, self._ref_index(ref))
        response = requests.get(url, auth=self._auth)
        response.raise_for_status()
        the_json = response.json()
        if not the_json["revisions"]:
            raise RecipeNotFoundException(ref)
        tmp = [_RevisionEntry(i["revision"], i["time"]) for i in the_json["revisions"]]
        return tmp

    def get_package_revisions_references(self, pref):
        time.sleep(0.1)  # Index appears to not being updated immediately
        url = "{}/{}".format(self._repo_url, self._pref_index(pref))
        response = requests.get(url, auth=self._auth)
        response.raise_for_status()
        the_json = response.json()
        if not the_json["revisions"]:
            raise PackageNotFoundException(pref)
        tmp = [_RevisionEntry(i["revision"], i["time"]) for i in the_json["revisions"]]
        return tmp

    def get_last_revision(self, ref):
        revisions = self.get_recipe_revisions_references(ref)
        return revisions[0]

    def get_last_package_revision(self, ref):
        revisions = self.get_package_revisions_references(ref)
        return revisions[0]


class ArtifactoryServer(object):

    def __init__(self, *args, **kwargs):
        self._user = ARTIFACTORY_DEFAULT_USER
        self._password = ARTIFACTORY_DEFAULT_PASSWORD
        self._url = ARTIFACTORY_DEFAULT_URL
        self._repo_name = "conan_{}".format(str(uuid.uuid4()).replace("-", ""))
        self.create_repository()
        self.server_store = _ArtifactoryServerStore(self.repo_url, self._user, self._password)

    @property
    def _auth(self):
        return self._user, self._password

    @property
    def repo_url(self):
        return "{}/{}".format(self._url, self._repo_name)

    @property
    def repo_api_url(self):
        return "{}/api/conan/{}".format(self._url, self._repo_name)

    def recipe_revision_time(self, ref):
        revs = self.server_store.get_recipe_revisions_references(ref)
        for r in revs:
            if r.revision == ref.revision:
                return r.time
        return None

    def package_revision_time(self, pref):
        revs = self.server_store.get_package_revisions_references(pref)
        for r in revs:
            if r.revision == pref.revision:
                return r.time
        return None

    def create_repository(self):
        url = "{}/api/repositories/{}".format(self._url, self._repo_name)
        config = {"key": self._repo_name, "rclass": "local", "packageType": "conan"}
        ret = requests.put(url, auth=self._auth, json=config)
        ret.raise_for_status()

    def package_exists(self, pref):
        try:
            revisions = self.server_store.get_package_revisions_references(pref)
            if pref.revision:
                for r in revisions:
                    if pref.revision == r.revision:
                        return True
                return False
            return True
        except Exception:  # When resolves the latest and there is no package
            return False

    def recipe_exists(self, ref):
        try:
            revisions = self.server_store.get_recipe_revisions_references(ref)
            if ref.revision:
                for r in revisions:
                    if ref.revision == r.revision:
                        return True
                return False
            return True
        except Exception:  # When resolves the latest and there is no package
            return False
