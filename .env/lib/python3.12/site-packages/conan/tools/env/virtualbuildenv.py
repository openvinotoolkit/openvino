from conan.internal import check_duplicated_generator
from conan.tools.env import Environment
from conan.tools.env.virtualrunenv import runenv_from_cpp_info


class VirtualBuildEnv:
    """ Calculates the environment variables of the build time context and produces a conanbuildenv
        .bat or .sh script
    """

    def __init__(self, conanfile, auto_generate=False):
        self._buildenv = None
        self._conanfile = conanfile
        if not auto_generate:
            self._conanfile.virtualbuildenv = False
        self.basename = "conanbuildenv"
        self.configuration = None
        self.arch = None

    @property
    def _filename(self):
        if not self.configuration:
            # TODO: Make this use the settings_build
            configuration = self._conanfile.settings.get_safe("build_type")
            configuration = configuration.lower() if configuration else None
        else:
            configuration = self.configuration
        if not self.arch:
            arch = self._conanfile.settings.get_safe("arch")
            arch = arch.lower() if arch else None
        else:
            arch = self.arch
        f = self.basename
        if configuration:
            f += "-" + configuration.replace(".", "_")
        if arch:
            f += "-" + arch.replace(".", "_").replace("|", "_")
        return f

    def environment(self):
        """
        Returns an ``Environment`` object containing the environment variables of the build context.

        :return: an ``Environment`` object instance containing the obtained variables.
        """

        if self._buildenv is None:
            self._buildenv = Environment()
        else:
            return self._buildenv

        # Top priority: profile
        profile_env = self._conanfile.buildenv
        self._buildenv.compose_env(profile_env)

        build_requires = self._conanfile.dependencies.build.topological_sort
        for require, build_require in reversed(build_requires.items()):
            if require.direct:  # Only buildenv_info from direct deps is propagated
                # higher priority, explicit buildenv_info
                if build_require.buildenv_info:
                    self._buildenv.compose_env(build_require.buildenv_info)
            # Lower priority, the runenv of all transitive "requires" of the build requires
            if build_require.runenv_info:
                self._buildenv.compose_env(build_require.runenv_info)
            # Then the implicit
            os_name = self._conanfile.settings_build.get_safe("os")
            self._buildenv.compose_env(runenv_from_cpp_info(build_require, os_name))

        # Requires in host context can also bring some direct buildenv_info
        host_requires = self._conanfile.dependencies.host.topological_sort
        for require in reversed(host_requires.values()):
            if require.buildenv_info:
                self._buildenv.compose_env(require.buildenv_info)

        return self._buildenv

    def vars(self, scope="build"):
        """
        :param scope: Scope to be used.
        :return: An ``EnvVars`` instance containing the computed environment variables.
        """
        return self.environment().vars(self._conanfile, scope=scope)

    def generate(self, scope="build"):
        """
        Produces the launcher scripts activating the variables for the build context.

        :param scope: Scope to be used.
        """
        check_duplicated_generator(self, self._conanfile)
        build_env = self.environment()
        build_env.vars(self._conanfile, scope=scope).save_script(self._filename)
