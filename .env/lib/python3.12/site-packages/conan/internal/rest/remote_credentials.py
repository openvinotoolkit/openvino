import json
import os
import platform

from jinja2 import Template

from conan.api.input import UserInput
from conan.internal.cache.home_paths import HomePaths

from conan.internal.loader import load_python_file
from conan.api.output import ConanOutput
from conan.internal.errors import scoped_traceback
from conan.errors import ConanException
from conan.internal.util.files import load


class RemoteCredentials:
    def __init__(self, cache_folder, global_conf):
        self._global_conf = global_conf
        self._urls = {}
        auth_plugin_path = HomePaths(cache_folder).auth_remote_plugin_path
        self._auth_remote_plugin = _load_auth_remote_plugin(auth_plugin_path)
        creds_path = os.path.join(cache_folder, "credentials.json")
        if not os.path.exists(creds_path):
            return
        try:
            template = Template(load(creds_path))
            content = template.render({"platform": platform, "os": os})
            content = json.loads(content)

            self._urls = {credentials["remote"]: {"user": credentials["user"],
                                                  "password": credentials["password"]}
                          for credentials in content["credentials"]}
        except Exception as e:
            raise ConanException(f"Error loading 'credentials.json' {creds_path}: {repr(e)}")

    def auth(self, remote, user=None):
        # First get the auth_remote_plugin
        if self._auth_remote_plugin is not None:
            try:
                plugin_user, plugin_password = self._auth_remote_plugin(remote, user=user)
            except Exception as e:
                msg = f"Error while processing 'auth_remote.py' plugin"
                msg = scoped_traceback(msg, e, scope="/extensions/plugins")
                raise ConanException(msg)
            if plugin_user and plugin_password:
                return plugin_user, plugin_password, False

        # Then prioritize the cache "credentials.json" file
        creds = self._urls.get(remote.name)
        if creds is not None:
            try:
                return creds["user"], creds["password"], False
            except KeyError as e:
                raise ConanException(f"Authentication error, wrong credentials.json: {e}")

        # Then, check environment definition
        env_user, env_passwd = self._get_env(remote.name, user)
        if env_passwd is not None:
            if env_user is None:
                raise ConanException("Found password in env-var, but not defined user")
            return env_user, env_passwd, False

        # If not found, then interactive prompt
        ui = UserInput(self._global_conf.get("core:non_interactive", check_type=bool))
        input_user, input_password = ui.request_login(remote.name, user)
        return input_user, input_password, True

    @staticmethod
    def _get_env(remote, user):
        """
        Try get creds from env-vars
        """
        remote = remote.replace("-", "_").upper()
        if user is None:
            user = os.getenv(f"CONAN_LOGIN_USERNAME_{remote}") or os.getenv("CONAN_LOGIN_USERNAME")
            if user:
                ConanOutput().info("Got username '%s' from environment" % user)
        passwd = os.getenv(f"CONAN_PASSWORD_{remote}") or os.getenv("CONAN_PASSWORD")
        if passwd:
            ConanOutput().info("Got password '******' from environment")
        return user, passwd


def _load_auth_remote_plugin(auth_remote_plugin_path):
    if os.path.exists(auth_remote_plugin_path):
        mod, _ = load_python_file(auth_remote_plugin_path)
        return getattr(mod, "auth_remote_plugin", None)
