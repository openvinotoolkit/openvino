import fnmatch
import json
import os
from collections import OrderedDict
from urllib.parse import urlparse

from conan.api.model import Remote, LOCAL_RECIPES_INDEX
from conan.api.output import ConanOutput
from conan.internal.cache.home_paths import HomePaths
from conan.internal.conan_app import ConanBasicApp
from conan.internal.rest.remote_credentials import RemoteCredentials
from conan.internal.rest.rest_client_local_recipe_index import add_local_recipes_index_remote, \
    remove_local_recipes_index_remote
from conan.internal.api.remotes.localdb import LocalDB
from conan.errors import ConanException
from conan.internal.util.files import save, load

CONAN_CENTER_REMOTE_NAME = "conancenter"


class RemotesAPI:
    """ The ``RemotesAPI`` manages the definition of remotes, contained in the "remotes.json" file
    in the Conan home, supporting addition, removal, update, rename, enable, disable of remotes.
    These operations do not contact the servers or check their existence at all. If they are not
    available, they will fail later when used.

    The ``user_xxx`` methods perform authentication related tasks, and some of them will contact
    the servers to perform such authentication
    """

    def __init__(self, conan_api, api_helpers):
        # This method is private, the subapi is not instantiated by users
        self._conan_api = conan_api
        self._api_helpers = api_helpers
        self._home_folder = conan_api.home_folder
        self._remotes_file = HomePaths(self._home_folder).remotes_path

    def list(self, pattern=None, only_enabled=True):
        """
        Obtain a list of :ref:`Remote <conan.api.model.Remote>` objects matching the pattern.

        :param pattern: ``None``, single ``str`` or list of ``str``. If it is ``None``,
          all remotes will be returned (equivalent to ``pattern="*"``).
        :param only_enabled: boolean, by default return only enabled remotes
        :return: A list of :ref:`Remote <conan.api.model.Remote>` objects

        """
        remotes = _load(self._remotes_file)
        if only_enabled:
            remotes = [r for r in remotes if not r.disabled]
        if pattern:
            remotes = _filter(remotes, pattern, only_enabled)
        return remotes

    def disable(self, pattern):
        """
        Disable all remotes matching ``pattern``

        :param pattern: single ``str`` or list of ``str``. If the pattern is an exact name without
          wildcards like "*" and no remote is found matching that exact name, it will raise an error.
        :return: the list of disabled :ref:`Remote <conan.api.model.Remote>` objects  (even if they
          were already disabled)
        """
        remotes = _load(self._remotes_file)
        disabled = _filter(remotes, pattern, only_enabled=False)
        result = []
        if disabled:
            for r in disabled:
                r.disabled = True
                result.append(r)
            _save(self._remotes_file, remotes)
        return result

    def enable(self, pattern):
        """
        Enable all remotes matching ``pattern``.

        :param pattern: single ``str`` or list of ``str``. If the pattern is an exact name without
          wildcards like "*" and no remote is found matching that exact name, it will raise an error.
        :return: the list of enabled :ref:`Remote <conan.api.model.Remote>` objects (even if they
          were already enabled)
        """
        remotes = _load(self._remotes_file)
        enabled = _filter(remotes, pattern, only_enabled=False)
        result = []
        if enabled:
            for r in enabled:
                r.disabled = False
                result.append(r)
            _save(self._remotes_file, remotes)
        return result

    def get(self, remote_name):
        """
        Obtain a :ref:`Remote <conan.api.model.Remote>` object

        :param remote_name: the exact name of the remote to be returned
        :return: the :ref:`Remote <conan.api.model.Remote>` object, or raise an Exception if the
          remote does not exist.
        """
        remotes = _load(self._remotes_file)
        try:
            return {r.name: r for r in remotes}[remote_name]
        except KeyError:
            raise ConanException(f"Remote '{remote_name}' doesn't exist")

    def add(self, remote: Remote, force=False, index=None):
        """
        Add a new :ref:`Remote <conan.api.model.Remote>` object to the existing ones


        :param remote: a :ref:`Remote <conan.api.model.Remote>` object to be added
        :param force: do not fail if the remote already exist (but default it fails)
        :param index: if not defined, the new remote will be last one. Pass an integer to insert
          the remote in that position instead of the last one
        """
        add_local_recipes_index_remote(self._home_folder, remote)
        remotes = _load(self._remotes_file)
        if remote.remote_type != LOCAL_RECIPES_INDEX:
            _validate_url(remote.url)
        current = {r.name: r for r in remotes}.get(remote.name)
        if current:  # same name remote existing!
            if not force:
                raise ConanException(f"Remote '{remote.name}' already exists in remotes "
                                     "(use --force to continue)")
            ConanOutput().warning(f"Remote '{remote.name}' already exists in remotes")
            if current.url != remote.url:
                ConanOutput().warning("Updating existing remote with new url")

        _check_urls(remotes, remote.url, force, current)
        if index is None:  # append or replace in place
            d = {r.name: r for r in remotes}
            d[remote.name] = remote
            remotes = list(d.values())
        else:
            remotes = [r for r in remotes if r.name != remote.name]
            remotes.insert(index, remote)
        _save(self._remotes_file, remotes)

    def remove(self, pattern):
        """
        Remove the remotes matching the ``pattern``

        :param pattern: single ``str`` or list of ``str``. If the pattern is an exact name without
          wildcards like "*" and no remote is found matching that exact name, it will raise an error.
        :return: The list of removed :ref:`Remote <conan.api.model.Remote>` objects
        """
        remotes = _load(self._remotes_file)
        removed = _filter(remotes, pattern, only_enabled=False)
        remotes = [r for r in remotes if r not in removed]
        _save(self._remotes_file, remotes)
        localdb = LocalDB(self._home_folder)
        for remote in removed:
            remove_local_recipes_index_remote(self._home_folder, remote)
            localdb.clean(remote_url=remote.url)
        return removed

    def update(self, remote_name: str, url=None, secure=None, disabled=None, index=None,
               allowed_packages=None, recipes_only=None):
        """
        Update an existing remote

        :param remote_name: The name of the remote to update, must exist
        :param url: optional url to update, if not defined it will not be updated
        :param secure:  optional ssl secure connection to update
        :param disabled: optional disabled state
        :param index:  optional integer to change the order of the remote
        :param allowed_packages: optional list of packages allowed from this remote
        :param recipes_only: optional boolean to only allow recipe downloads from this remote,
            never package binaries
        """
        remotes = _load(self._remotes_file)
        try:
            remote = {r.name: r for r in remotes}[remote_name]
        except KeyError:
            raise ConanException(f"Remote '{remote_name}' doesn't exist")
        if url is not None:
            if remote.remote_type != LOCAL_RECIPES_INDEX:
                _validate_url(url)
            _check_urls(remotes, url, force=False, current=remote)
            remote.url = url
        if secure is not None:
            remote.verify_ssl = secure
        if disabled is not None:
            remote.disabled = disabled
        if allowed_packages is not None:
            remote.allowed_packages = allowed_packages
        if recipes_only is not None:
            remote.recipes_only = recipes_only

        if index is not None:
            remotes = [r for r in remotes if r.name != remote.name]
            remotes.insert(index, remote)
        _save(self._remotes_file, remotes)

    def rename(self, remote_name: str, new_name: str):
        """
        Change the name of an existing remote

        :param remote_name: The previous existing name
        :param new_name: The new name
        """
        remotes = _load(self._remotes_file)
        d = {r.name: r for r in remotes}
        if new_name in d:
            raise ConanException(f"Remote '{new_name}' already exists")
        try:
            d[remote_name].name = new_name
        except KeyError:
            raise ConanException(f"Remote '{remote_name}' doesn't exist")
        _save(self._remotes_file, remotes)

    def user_info(self, remote: Remote):
        # TODO: Review
        localdb = LocalDB(self._home_folder)
        user_info = {}
        user, token, _ = localdb.get_login(remote.url)
        user_info["name"] = remote.name
        user_info["user_name"] = user
        user_info["authenticated"] = True if token else False
        return user_info

    def user_login(self, remote: Remote, username: str, password: str):
        """
        Perform user authentication against the given remote with the provided username and password

        :param remote: a :ref:`Remote <conan.api.model.Remote>` object
        :param username: the user login as ``str``
        :param password: password ``str``
        """
        app = ConanBasicApp(self._conan_api)
        app.remote_manager.authenticate(remote, username, password)

    def login(self, remotes, username=None, password=None):
        creds = RemoteCredentials(self._conan_api.cache_folder, self._api_helpers.global_conf)

        ret = OrderedDict()
        for r in remotes:
            previous_info = self.user_info(r)

            if username is not None and password is not None:
                user, password = username, password
            else:
                user, password, _ = creds.auth(r, username)
                if username is not None and username != user:
                    raise ConanException(f"User '{username}' doesn't match user '{user}' in "
                                         f"credentials.json or environment variables")

            self.user_login(r, user, password)
            info = self.user_info(r)
            ret[r.name] = {"previous_info": previous_info, "info": info}
        return ret

    def user_logout(self, remote: Remote):
        """
        Logout from the given :ref:`Remote <conan.api.model.Remote>`

        :param remote: The :ref:`Remote <conan.api.model.Remote>` object to logout
        """
        localdb = LocalDB(self._home_folder)
        # The localdb only stores url + username + token, not remote name, so use URL as key
        localdb.clean(remote_url=remote.url)

    def user_set(self, remote: Remote, username):
        # TODO: Review
        localdb = LocalDB(self._home_folder)
        if username == "":
            username = None
        localdb.store(username, token=None, refresh_token=None, remote_url=remote.url)

    def user_auth(self, remote: Remote, with_user=False, force=False):
        # TODO: Review
        localdb = LocalDB(self._home_folder)
        app = ConanBasicApp(self._conan_api)
        if with_user:
            user, token, _ = localdb.get_login(remote.url)
            if not user:
                var_name = f"CONAN_LOGIN_USERNAME_{remote.name.replace('-', '_').upper()}"
                user = os.getenv(var_name, None) or os.getenv("CONAN_LOGIN_USERNAME", None)
            if not user:
                return
        app.remote_manager.check_credentials(remote, force)
        user, token, _ = localdb.get_login(remote.url)
        return user


def _load(remotes_file):
    if not os.path.exists(remotes_file):
        remote = Remote(CONAN_CENTER_REMOTE_NAME, "https://center2.conan.io", True, False)
        _save(remotes_file, [remote])
        return [remote]

    try:
        data = json.loads(load(remotes_file))
    except Exception as e:
        raise ConanException(f"Error loading JSON remotes file '{remotes_file}': {e}")
    result = []
    for r in data.get("remotes", []):
        remote = Remote(r["name"], r["url"], r["verify_ssl"], r.get("disabled", False),
                        r.get("allowed_packages"), r.get("remote_type"),
                        r.get("recipes_only", False))
        result.append(remote)
    return result


def _save(remotes_file, remotes):
    remote_list = []
    for r in remotes:
        remote = {"name": r.name, "url": r.url, "verify_ssl": r.verify_ssl}
        if r.disabled:
            remote["disabled"] = True
        if r.allowed_packages:
            remote["allowed_packages"] = r.allowed_packages
        if r.remote_type:
            remote["remote_type"] = r.remote_type
        if r.recipes_only:
            remote["recipes_only"] = r.recipes_only
        remote_list.append(remote)
    # This atomic replace avoids a corrupted remotes.json file if this is killed during the process
    save(remotes_file + ".tmp", json.dumps({"remotes": remote_list}, indent=True))
    os.replace(remotes_file + ".tmp", remotes_file)


def _filter(remotes, pattern, only_enabled=True):
    filtered_remotes = []
    patterns = [pattern] if isinstance(pattern, str) else pattern
    for p in patterns:
        is_match = False
        for remote in remotes:
            if fnmatch.fnmatch(remote.name, p):
                is_match = True
                if remote not in filtered_remotes:
                    filtered_remotes.append(remote)
        if not is_match:
            if "*" in p or "?" in p:
                if only_enabled:
                    raise ConanException(
                        f"Remotes for pattern '{p}' can't be found or are disabled")
            else:
                raise ConanException(f"Remote '{p}' can't be found or is disabled")
    return filtered_remotes


def _validate_url(url):
    """ Check if URL contains protocol and address
    :param url: URL to be validated
    """
    out = ConanOutput()
    if url:
        if url.startswith("https://conan.io/center"):
            raise ConanException("Wrong ConanCenter remote URL. You are adding the web "
                                 "https://conan.io/center the correct remote API is "
                                 "https://center2.conan.io")
        address = urlparse(url)
        if not all([address.scheme, address.netloc]):
            out.warning(f"The URL '{url}' is invalid. It must contain scheme and hostname.")
    else:
        out.warning("The URL is empty. It must contain scheme and hostname.")


def _check_urls(remotes, url, force, current):
    # The remote name doesn't exist
    for r in remotes:
        if r is not current and r.url == url:
            msg = f"Remote url already existing in remote '{r.name}'. " \
                  f"Having different remotes with same URL is not recommended."
            if not force:
                raise ConanException(msg + " Use '--force' to override.")
            else:
                ConanOutput().warning(msg + " Adding duplicated remote url because '--force'.")
