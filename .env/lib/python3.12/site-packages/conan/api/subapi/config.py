import os

from conan.api.output import ConanOutput

from conan.internal.cache.home_paths import HomePaths
from conan.internal.conan_app import ConanApp
from conan.internal.graph.graph import CONTEXT_HOST, RECIPE_VIRTUAL, Node
from conan.internal.graph.graph_builder import DepsGraphBuilder
from conan.internal.graph.profile_node_definer import consumer_definer
from conan.errors import ConanException

from conan.internal.model.conanconfig import loadconanconfig, saveconanconfig, loadconanconfig_yml
from conan.internal.model.conf import BUILT_IN_CONFS
from conan.internal.model.pkg_type import PackageType
from conan.api.model import RecipeReference, Remote
from conan.internal.util.files import rmdir, remove


class ConfigAPI:
    """ This API provides methods to manage the Conan configuration in the Conan home folder.
    It allows installing configurations from various sources, retrieving global configuration
    values, and listing available configurations. It also provides methods to clean the
    Conan home folder, resetting it to a clean state.
    """

    def __init__(self, conan_api, helpers):
        self._conan_api = conan_api
        self._helpers = helpers

    def home(self):
        """ return the current Conan home folder containing the configuration files like
        remotes, settings, profiles, and the packages cache. It is provided for debugging
        purposes. Recall that it is not allowed to write, modify or remove packages in the
        packages cache, and that to automate tasks that uses packages from the cache Conan
        provides mechanisms like deployers or custom commands.
        """
        return self._conan_api.cache_folder

    def install(self, path_or_url, verify_ssl, config_type=None, args=None,
                source_folder=None, target_folder=None):
        """ install Conan configuration from a git repo, from a zip file in an http server
        or a local folder
        """
        from conan.internal.api.config.config_installer import configuration_install
        cache_folder = self._conan_api.cache_folder
        requester = self._helpers.requester
        configuration_install(cache_folder, requester, path_or_url, verify_ssl,
                              config_type=config_type, args=args,
                              source_folder=source_folder, target_folder=target_folder)
        self._conan_api.reinit()

    def install_package(self, require, lockfile=None, force=False, remotes=None, profile=None):
        ConanOutput().warning("The 'conan config install-pkg' is experimental",
                              warn_tag="experimental")
        require = RecipeReference.loads(require)
        required_pkgs = self.fetch_packages([require], lockfile, remotes, profile)
        installed_refs = self._install_pkgs(required_pkgs, force)
        self._conan_api.reinit()
        return installed_refs

    @staticmethod
    def load_conanconfig(path, remotes):
        if os.path.isdir(path):
            path = os.path.join(path, "conanconfig.yml")
        requested_requires, urls = loadconanconfig_yml(path)
        if urls:
            new_remotes = [Remote(f"config_install_url{'_' + str(i)}", url=url)
                           for i, url in enumerate(urls)]
            remotes = remotes or []
            remotes += new_remotes
        return requested_requires, remotes

    def install_conanconfig(self, path, lockfile=None, force=False, remotes=None, profile=None):
        ConanOutput().warning("The 'conan config install-pkg' is experimental",
                              warn_tag="experimental")
        requested_requires, remotes = self.load_conanconfig(path, remotes)
        required_pkgs = self.fetch_packages(requested_requires, lockfile, remotes, profile)
        installed_refs = self._install_pkgs(required_pkgs, force)
        self._conan_api.reinit()
        return installed_refs

    def _install_pkgs(self, required_pkgs, force):
        out = ConanOutput()
        out.title("Configuration packages to install")
        config_version_file = HomePaths(self._conan_api.home_folder).config_version_path
        if not os.path.exists(config_version_file):
            config_versions = []
        else:
            ConanOutput().info(f"Reading existing config-versions file: {config_version_file}")
            config_versions = loadconanconfig(config_version_file)
        config_versions_dict = {r.name: r for r in config_versions}
        if len(config_versions_dict) < len(config_versions):
            raise ConanException("There are multiple requirements for the same package "
                                 f"with different versions: {config_version_file}")

        new_config = config_versions_dict.copy()
        for required_pkg in required_pkgs:
            new_config.pop(required_pkg.ref.name, None)  # To ensure new order
            new_config[required_pkg.ref.name] = required_pkg.ref
        final_config_refs = [r for r in new_config.values()]

        prev_refs = "\n\t".join(repr(r) for r in config_versions)
        out.info(f"Previously installed configuration packages:\n\t{prev_refs}")

        new_refs = "\n\t".join(r.repr_notime() for r in final_config_refs)
        out.info(f"New configuration packages to install:\n\t{new_refs}")

        if list(config_versions_dict) == list(new_config)[:len(config_versions_dict)]:
            # There is no conflict in order, can be done safely
            if final_config_refs == config_versions:
                if force:
                    out.warning("The requested configurations are identical to the already "
                                "installed ones, but forcing re-installation because --force")
                    to_install = required_pkgs
                else:
                    out.info("The requested configurations are identical to the already "
                             "installed ones, skipping re-installation")
                    to_install = []
            else:
                out.info("Installing new or updating configuration packages")
                to_install = required_pkgs
        else:
            # Change in order of existing configuration
            if force:
                out.warning("Installing these configuration packages will break the "
                            "existing order, with possible side effects. "
                            "Forcing the installation because --force was defined", warn_tag="risk")
                to_install = required_pkgs
            else:
                msg = ("Installing these configuration packages will break the "
                       "existing order, with possible side effects, like breaking 'package_ids'.\n"
                       "If you still want to enforce this configuration you can:\n"
                       "   Use 'conan config clean' first to fully reset your configuration.\n"
                       "   Or use 'conan config install-pkg --force' to force installation.")
                raise ConanException(msg)

        out.title("Installing configuration from packages")
        # install things and update the Conan cache "config_versions.json" file
        from conan.internal.api.config.config_installer import configuration_install
        cache_folder = self._conan_api.cache_folder
        requester = self._helpers.requester
        for pkg in to_install:
            out.info(f"Installing configuration from {pkg.ref}")
            configuration_install(cache_folder, requester, uri=pkg.conanfile.package_folder,
                                  verify_ssl=False, config_type="dir",
                                  ignore=["conaninfo.txt", "conanmanifest.txt"])

        saveconanconfig(config_version_file, final_config_refs)
        return final_config_refs

    def fetch_packages(self, refs, lockfile=None, remotes=None, profile=None):
        """ install configuration stored inside a Conan package
        The installation of configuration will reinitialize the full ConanAPI
        """
        conan_api = self._conan_api
        remotes = conan_api.remotes.list() if remotes is None else remotes
        profile_host = profile_build = profile or conan_api.profiles.get_profile([])

        app = ConanApp(self._conan_api)

        ConanOutput().title("Fetching requested configuration packages")
        result = []
        for ref in refs:
            # Computation of a very simple graph that requires "ref"
            # Need to convert input requires to RecipeReference
            conanfile = app.loader.load_virtual(requires=[ref])
            consumer_definer(conanfile, profile_host, profile_build)
            root_node = Node(ref=None, conanfile=conanfile, context=CONTEXT_HOST,
                             recipe=RECIPE_VIRTUAL)
            root_node.is_conf = True
            update = ["*"]
            builder = DepsGraphBuilder(app.proxy, app.loader, app.range_resolver, app.cache, remotes,
                                       update, update, self._helpers.global_conf)
            deps_graph = builder.load_graph(root_node, profile_host, profile_build, lockfile)

            # Basic checks of the package: correct package_type and no-dependencies
            deps_graph.report_graph_error()
            pkg = deps_graph.root.edges[0].dst
            ConanOutput().info(f"Configuration from package: {pkg}")
            if pkg.conanfile.package_type is not PackageType.CONF:
                raise ConanException(f'{pkg.conanfile} is not of package_type="configuration"')
            if pkg.edges:
                raise ConanException(f"Configuration package {pkg.ref} cannot have dependencies")

            # The computation of the "package_id" and the download of the package is done as usual
            # By default we allow all remotes, and build_mode=None, always updating
            conan_api.graph.analyze_binaries(deps_graph, None, remotes, update=update,
                                             lockfile=lockfile)
            conan_api.install.install_binaries(deps_graph=deps_graph, remotes=remotes)
            result.append(pkg)
        return result

    def get(self, name, default=None, check_type=None):
        """ get the value of a global.conf item
        """
        return self._helpers.global_conf.get(name, default=default, check_type=check_type)

    def show(self, pattern) -> dict:
        """ get the values of global.conf for those configurations that matches the pattern
        """
        return self._helpers.global_conf.show(pattern)

    @staticmethod
    def conf_list():
        """ list all the available built-in configurations
        """
        return BUILT_IN_CONFS.copy()

    def clean(self):
        """ reset the Conan home folder to a clean state, removing all the user
        custom configuration, custom files, and resetting modified files
        """
        contents = os.listdir(self.home())
        packages_folder = (self._helpers.global_conf.get("core.cache:storage_path") or
                           os.path.join(self.home(), "p"))
        for content in contents:
            content_path = os.path.join(self.home(), content)
            if content_path == packages_folder:
                continue
            ConanOutput().debug(f"Removing {content_path}")
            if os.path.isdir(content_path):
                rmdir(content_path)
            else:
                remove(content_path)
        self._conan_api.reinit()
        # CHECK: This also generates a remotes.json that is not there after a conan profile show?
        self._conan_api.migrate()

    @property
    def settings_yml(self):
        """ Get the contents of the settings.yml and user_settings.yml files,
            which define the possible values for settings.

            Note that this is different from the settings present in a conanfile,
            which represent the actual values for a specific package, while this
            property represents the possible values for each setting.

            :returns: A read-only object representing the settings scheme, with a
                ``possible_values()`` method that returns a dictionary with the possible values for each setting,
                and a ``fields`` property that returns an ordered list with the fields of each setting.
                Note that it's possible to access nested settings using attribute access,
                such as ``settings_yml.compiler.possible_values()``.
        """

        class SettingsYmlInterface:
            def __init__(self, settings):
                self._settings = settings

            def possible_values(self):
                """ returns a dict with the possible values for each setting """
                return self._settings.possible_values()

            @property
            def fields(self):
                """ returns a dict with the fields of each setting """
                return self._settings.fields

            def __getattr__(self, item):
                return SettingsYmlInterface(getattr(self._settings, item))

            def __str__(self):
                return str(self._settings)

        return SettingsYmlInterface(self._helpers.settings_yml)
