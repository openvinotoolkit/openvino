import os

from conan.api.output import ConanOutput

_EXTENSIONS_FOLDER = "extensions"
_PLUGINS = "plugins"


class HomePaths:
    """ pure computing of paths in the home, not caching anything
    """
    def __init__(self, home_folder):
        self._home = home_folder

    @property
    def local_recipes_index_path(self):
        return os.path.join(self._home, ".local_recipes_index")

    @property
    def global_conf_path(self):
        return os.path.join(self._home, "global.conf")

    @property
    def deployers_path(self):
        deploy = os.path.join(self._home, _EXTENSIONS_FOLDER, "deploy")
        if os.path.exists(deploy):
            ConanOutput().warning("Use 'deployers' cache folder for deployers instead of 'deploy'",
                                  warn_tag="deprecated")
            return deploy
        return os.path.join(self._home, _EXTENSIONS_FOLDER, "deployers")

    @property
    def custom_generators_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, "generators")

    @property
    def hooks_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, "hooks")

    @property
    def wrapper_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, _PLUGINS, "cmd_wrapper.py")

    @property
    def profiles_path(self):
        return os.path.join(self._home, "profiles")

    @property
    def profile_plugin_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, _PLUGINS, "profile.py")

    @property
    def auth_remote_plugin_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, _PLUGINS, "auth_remote.py")

    @property
    def auth_source_plugin_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, _PLUGINS, "auth_source.py")

    @property
    def sign_plugin_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, _PLUGINS, "sign", "sign.py")

    @property
    def remotes_path(self):
        return os.path.join(self._home, "remotes.json")

    @property
    def providers_path(self):
        return os.path.join(self._home, "audit_providers.json")

    @property
    def compatibility_plugin_path(self):
        return os.path.join(self._home, _EXTENSIONS_FOLDER, _PLUGINS, "compatibility")

    @property
    def default_sources_backup_folder(self):
        return os.path.join(self._home, "sources")

    @property
    def settings_path(self):
        return os.path.join(self._home, "settings.yml")

    @property
    def settings_path_user(self):
        return os.path.join(self._home, "settings_user.yml")

    @property
    def config_version_path(self):
        return os.path.join(self._home, "config_version.json")
