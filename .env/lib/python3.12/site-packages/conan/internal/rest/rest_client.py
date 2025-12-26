from conan.internal import REVISIONS
from conan.internal.rest.rest_client_v2 import RestV2Methods
from conan.errors import ConanException

CHECKSUM_DEPLOY = "checksum_deploy"  # capability


class RestApiClient:
    """
        Rest Api Client for handle remote.
    """

    def __init__(self, remote, token, requester, config):
        self._token = token
        self._remote_url = remote.url
        self._requester = requester
        self._verify_ssl = remote.verify_ssl
        self._config = config
        self._remote = remote

    def _capable(self, capability, user=None, password=None):
        # Caching of capabilities per-remote
        capabilities = getattr(self._remote, "_capabilities", None)
        if capabilities is None:
            tmp = RestV2Methods(self._remote_url, self._token,
                                self._requester, self._config, self._verify_ssl)
            capabilities = tmp.server_capabilities(user, password)
            setattr(self._remote, "_capabilities", capabilities)
        return capability in capabilities

    def _get_api(self):
        revisions = self._capable(REVISIONS)

        if not revisions:
            raise ConanException("The remote doesn't support revisions. "
                                 "Conan 2.0 is no longer compatible with "
                                 "remotes that don't accept revisions.")
        checksum_deploy = self._capable(CHECKSUM_DEPLOY)
        return RestV2Methods(self._remote_url, self._token,
                             self._requester, self._config, self._verify_ssl,
                             checksum_deploy)

    def get_recipe(self, ref, dest_folder, metadata, only_metadata):
        return self._get_api().get_recipe(ref, dest_folder, metadata, only_metadata)

    def get_recipe_sources(self, ref, dest_folder):
        return self._get_api().get_recipe_sources(ref, dest_folder)

    def get_package(self, pref, dest_folder, metadata, only_metadata):
        return self._get_api().get_package(pref, dest_folder, metadata, only_metadata)

    def upload_recipe(self, ref, files_to_upload):
        return self._get_api().upload_recipe(ref, files_to_upload)

    def upload_package(self, pref, files_to_upload):
        return self._get_api().upload_package(pref, files_to_upload)

    def authenticate(self, user, password):
        # BYPASS capabilities, in case v1/ping is protected
        api_v2 = RestV2Methods(self._remote_url, self._token,
                               self._requester, self._config, self._verify_ssl)
        token = api_v2.authenticate(user, password)
        return token

    def check_credentials(self, force_auth=False):
        return self._get_api().check_credentials(force_auth)

    def search(self, pattern=None, ignorecase=True):
        return self._get_api().search(pattern, ignorecase)

    def search_packages(self, reference, list_only=False):
        return self._get_api().search_packages(reference, list_only)

    def remove_recipe(self, ref):
        return self._get_api().remove_recipe(ref)

    def remove_all_packages(self, ref):
        return self._get_api().remove_all_packages(ref)

    def remove_packages(self, prefs):
        return self._get_api().remove_packages(prefs)

    def get_recipe_revisions_references(self, ref):
        return self._get_api().get_recipe_revisions_references(ref)

    def get_package_revisions_references(self, pref):
        return self._get_api().get_package_revisions_references(pref)

    def get_latest_recipe_reference(self, ref):
        return self._get_api().get_latest_recipe_reference(ref)

    def get_latest_package_reference(self, pref, headers):
        return self._get_api().get_latest_package_reference(pref, headers=headers)

    def get_recipe_revision_reference(self, ref):
        return self._get_api().get_recipe_revision_reference(ref)

    def get_package_revision_reference(self, pref):
        return self._get_api().get_package_revision_reference(pref)
