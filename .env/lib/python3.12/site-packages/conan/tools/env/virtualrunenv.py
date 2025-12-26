import os

from conan.internal import check_duplicated_generator
from conan.tools.env import Environment


def runenv_from_cpp_info(dep, os_name):
    """ return an Environment deducing the runtime information from a cpp_info
    """
    dyn_runenv = Environment()
    cpp_info = dep.cpp_info.aggregated_components()

    def _prepend_path(envvar, paths):
        existing = [p for p in paths if os.path.exists(p)] if paths else None
        if existing:
            dyn_runenv.prepend_path(envvar, existing)

    _prepend_path("PATH", cpp_info.bindirs)
    # If it is a build_require this will be the build-os, otherwise it will be the host-os
    if os_name and not os_name.startswith("Windows"):
        _prepend_path("LD_LIBRARY_PATH", cpp_info.libdirs)
        _prepend_path("DYLD_LIBRARY_PATH", cpp_info.libdirs)
        _prepend_path("DYLD_FRAMEWORK_PATH", cpp_info.frameworkdirs)
    return dyn_runenv


class VirtualRunEnv:
    """ Calculates the environment variables of the runtime context and produces a conanrunenv
        .bat or .sh script
    """

    def __init__(self, conanfile, auto_generate=False):
        """

        :param conanfile:  The current recipe object. Always use ``self``.
        """
        self._runenv = None
        self._conanfile = conanfile
        if not auto_generate:
            self._conanfile.virtualrunenv = False
        self.basename = "conanrunenv"
        self.configuration = conanfile.settings.get_safe("build_type")
        if self.configuration:
            self.configuration = self.configuration.lower()
        self.arch = conanfile.settings.get_safe("arch")
        if self.arch:
            self.arch = self.arch.lower()

    @property
    def _filename(self):
        f = self.basename
        if self.configuration:
            f += "-" + self.configuration.replace(".", "_")
        if self.arch:
            f += "-" + self.arch.replace(".", "_").replace("|", "_")
        return f

    def environment(self):
        """
        Returns an ``Environment`` object containing the environment variables of the run context.

        :return: an ``Environment`` object instance containing the obtained variables.
        """

        if self._runenv is None:
            self._runenv = Environment()
        else:
            return self._runenv

        # Top priority: profile
        profile_env = self._conanfile.runenv
        self._runenv.compose_env(profile_env)

        host_req = self._conanfile.dependencies.host
        test_req = self._conanfile.dependencies.test
        for require, dep in list(host_req.items()) + list(test_req.items()):
            if dep.runenv_info:
                self._runenv.compose_env(dep.runenv_info)
            if require.run:  # Only if the require is run (shared or application to be run)
                _os = self._conanfile.settings.get_safe("os")
                self._runenv.compose_env(runenv_from_cpp_info(dep, _os))

        return self._runenv

    def vars(self, scope="run"):
        """
        :param scope: Scope to be used.
        :return: An ``EnvVars`` instance containing the computed environment variables.
        """
        return self.environment().vars(self._conanfile, scope=scope)

    def generate(self, scope="run"):
        """
        Produces the launcher scripts activating the variables for the run context.

        :param scope: Scope to be used.
        """
        check_duplicated_generator(self, self._conanfile)
        run_env = self.environment()
        run_env.vars(self._conanfile, scope=scope).save_script(self._filename)
