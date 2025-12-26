import os
import subprocess
from pathlib import Path

from conan.api.output import ConanOutput, Color, LEVEL_QUIET
from conan.internal.subsystems import command_env_wrapper
from conan.errors import ConanException
from conan.internal.model.cpp_info import MockInfoProperty
from conan.internal.model.conf import Conf
from conan.internal.model.dependencies import ConanFileDependencies
from conan.internal.model.layout import Folders, Infos, Layouts
from conan.internal.model.options import Options
from conan.internal.model.requires import Requirements
from conan.internal.model.settings import Settings


class ConanFile:
    """
    The base class for all package recipes
    """

    # Reference
    name = None
    version = None  # Any str, can be "1.1" or whatever
    user = None
    channel = None

    # Metadata
    url = None  # The URL where this File is located, as github, to collaborate in package
    license = None
    author = None
    description = None
    topics = None
    homepage = None

    build_policy = None
    upload_policy = None

    exports = None
    exports_sources = None

    generators = []
    revision_mode = "hash"

    # Binary model: Settings and Options
    settings = None
    options = None
    default_options = None
    default_build_options = None
    package_type = None
    vendor = False
    languages = []
    implements = []

    provides = None
    deprecated = None

    win_bash = None
    win_bash_run = None  # For run scope

    _conan_is_consumer = False

    # #### Requirements
    requires = None
    tool_requires = None
    build_requires = None
    test_requires = None
    tested_reference_str = None

    no_copy_source = False
    recipe_folder = None

    # Package information
    cpp = None
    buildenv_info = None
    runenv_info = None
    conf_info = None
    generator_info = None
    conan_data = None

    def __init__(self, display_name=""):
        self.display_name = display_name
        # something that can run commands, as os.sytem

        self._conan_helpers = None
        from conan.tools.env import Environment
        self.buildenv_info = Environment()
        self.runenv_info = Environment()
        # At the moment only for build_requires, others will be ignored
        self.conf_info = Conf()
        self.info = None
        self._conan_buildenv = None  # The profile buildenv, will be assigned initialize()
        self._conan_runenv = None
        self._conan_node = None  # access to container Node object, to access info, context, deps...

        if isinstance(self.generators, str):
            self.generators = [self.generators]
        if isinstance(self.languages, str):
            self.languages = [self.languages]
        if not all(lang == "C" or lang == "C++" for lang in self.languages):
            raise ConanException("Only 'C' and 'C++' languages are allowed in 'languages' attribute")
        if isinstance(self.settings, str):
            self.settings = [self.settings]
        self.requires = Requirements(self.requires, self.build_requires, self.test_requires,
                                     self.tool_requires)

        self.options = Options(self.options or {}, self.default_options)

        if isinstance(self.topics, str):
            self.topics = [self.topics]
        if isinstance(self.provides, str):
            self.provides = [self.provides]

        # user declared variables
        self.user_info = MockInfoProperty("user_info")
        self.env_info = MockInfoProperty("env_info")
        self._conan_dependencies = None

        if not hasattr(self, "virtualbuildenv"):  # Allow the user to override it with True or False
            self.virtualbuildenv = True
        if not hasattr(self, "virtualrunenv"):  # Allow the user to override it with True or False
            self.virtualrunenv = True

        self.env_scripts = {}  # Accumulate the env scripts generated in order
        self.system_requires = {}  # Read only, internal {"apt": []}

        # layout() method related variables:
        self.folders = Folders()
        self.cpp = Infos()
        self.layouts = Layouts()

    def serialize(self):
        result = {}

        for a in ("name", "user", "channel", "url", "license",
                  "author", "description", "homepage", "build_policy", "upload_policy",
                  "revision_mode", "provides", "deprecated", "win_bash", "win_bash_run",
                  "default_options", "options_description"):
            v = getattr(self, a, None)
            result[a] = v

        result["version"] = str(self.version) if self.version is not None else None
        result["topics"] = list(self.topics) if self.topics is not None else None
        result["package_type"] = str(self.package_type)
        result["languages"] = self.languages

        settings = self.settings
        if settings is not None:
            result["settings"] = settings.serialize() if isinstance(settings, Settings) else list(settings)

        result["options"] = self.options.serialize()
        result["options_definitions"] = self.options.possible_values

        if self.generators is not None:
            result["generators"] = list(s.__name__ if isinstance(s, type) else s
                                        for s in self.generators)
        if self.license is not None:
            result["license"] = list(self.license) if not isinstance(self.license, str) else self.license

        result["requires"] = self.requires.serialize()

        if hasattr(self, "python_requires"):
            result["python_requires"] = self.python_requires.serialize()
        else:
            result["python_requires"] = None
        result["system_requires"] = self.system_requires

        result["recipe_folder"] = self.recipe_folder
        result["source_folder"] = self.source_folder
        result["build_folder"] = self.build_folder
        result["generators_folder"] = self.generators_folder
        result["package_folder"] = self.package_folder
        result["immutable_package_folder"] = self.immutable_package_folder

        result["cpp_info"] = self.cpp_info.serialize()
        result["conf_info"] = self.conf_info.serialize()
        result["label"] = self.display_name
        if self.info is not None:
            result["info"] = self.info.serialize()
        result["vendor"] = self.vendor
        if self.conan_data:
            result["conandata"] = self.conan_data
        return result

    @property
    def output(self):
        # an output stream (writeln, info, warn error)
        scope = self.display_name
        if not scope:
            scope = self.ref if self._conan_node else ""
        return ConanOutput(scope=scope)

    @property
    def context(self):
        return self._conan_node.context

    @property
    def subgraph(self):
        return self._conan_node.subgraph()

    @property
    def dependencies(self):
        # Caching it, this object is requested many times
        if self._conan_dependencies is None:
            self._conan_dependencies = ConanFileDependencies.from_node(self._conan_node)
        return self._conan_dependencies

    @property
    def ref(self):
        return self._conan_node.ref

    @property
    def pref(self):
        return self._conan_node.pref

    @property
    def buildenv(self):
        # Lazy computation of the package buildenv based on the profileone
        from conan.tools.env import Environment
        if not isinstance(self._conan_buildenv, Environment):
            self._conan_buildenv = self._conan_buildenv.get_profile_env(self.ref,
                                                                        self._conan_is_consumer)
        return self._conan_buildenv

    @property
    def runenv(self):
        # Lazy computation of the package runenv based on the profile one
        from conan.tools.env import Environment
        if not isinstance(self._conan_runenv, Environment):
            self._conan_runenv = self._conan_runenv.get_profile_env(self.ref,
                                                                    self._conan_is_consumer)
        return self._conan_runenv

    @property
    def cpp_info(self):
        """
        Same as using ``self.cpp.package`` in the ``layout()`` method. Use it if you need to read
        the ``package_folder`` to locate the already located artifacts.
        """
        return self.cpp.package

    @cpp_info.setter
    def cpp_info(self, value):
        self.cpp.package = value

    @property
    def source_folder(self):
        """
        The folder in which the source code lives. The path is built joining the base directory
        (a cache directory when running in the cache or the ``output folder`` when running locally)
        with the value of ``folders.source`` if declared in the ``layout()`` method.

        :return: A string with the path to the source folder.
        """
        return self.folders.source_folder

    @property
    def source_path(self) -> Path:
        self.output.warning(f"Use of 'source_path' is deprecated, please use 'source_folder' instead",
                            warn_tag="deprecated")
        assert self.source_folder is not None, "`source_folder` is `None`"
        return Path(self.source_folder)

    @property
    def export_sources_folder(self):
        """
        The value depends on the method you access it:

            - At ``source(self)``: Points to the base source folder (that means self.source_folder but
              without taking into account the ``folders.source`` declared in the ``layout()`` method).
              The declared `exports_sources` are copied to that base source folder always.
            - At ``exports_sources(self)``: Points to the folder in the cache where the export sources
              have to be copied.

        :return: A string with the mentioned path.
        """
        return self.folders.base_export_sources

    @property
    def export_sources_path(self) -> Path:
        self.output.warning(f"Use of 'export_sources_path' is deprecated, please use "
                            f"'export_sources_folder' instead", warn_tag="deprecated")
        assert self.export_sources_folder is not None, "`export_sources_folder` is `None`"
        return Path(self.export_sources_folder)

    @property
    def export_folder(self):
        return self.folders.base_export

    @property
    def export_path(self) -> Path:
        self.output.warning(f"Use of 'export_path' is deprecated, please use 'export_folder' instead",
                            warn_tag="deprecated")

        assert self.export_folder is not None, "`export_folder` is `None`"
        return Path(self.export_folder)

    @property
    def build_folder(self):
        """
        The folder used to build the source code. The path is built joining the base directory (a cache
        directory when running in the cache or the ``output folder`` when running locally) with
        the value of ``folders.build`` if declared in the ``layout()`` method.

        :return: A string with the path to the build folder.
        """
        return self.folders.build_folder

    @property
    def recipe_metadata_folder(self):
        return self.folders.recipe_metadata_folder

    @property
    def package_metadata_folder(self):
        return self.folders.package_metadata_folder

    @property
    def build_path(self) -> Path:
        self.output.warning(f"Use of 'build_path' is deprecated, please use 'build_folder' instead",
                            warn_tag="deprecated")
        assert self.build_folder is not None, "`build_folder` is `None`"
        return Path(self.build_folder)

    @property
    def package_folder(self):
        """
        The folder to copy the final artifacts for the binary package. In the local cache a package
        folder is created for every different package ID.

        :return: A string with the path to the package folder.
        """
        return self.folders.base_package

    @property
    def immutable_package_folder(self):
        return self.folders.immutable_package_folder

    @property
    def generators_folder(self):
        return self.folders.generators_folder

    @property
    def package_path(self) -> Path:
        self.output.warning(f"Use of 'package_path' is deprecated, please use 'package_folder' instead",
                            warn_tag="deprecated")

        assert self.package_folder is not None, "`package_folder` is `None`"
        return Path(self.package_folder)

    @property
    def generators_path(self) -> Path:
        self.output.warning(f"Use of 'generators_path' is deprecated, please use "
                            f"'generators_folder' instead", warn_tag="deprecated")
        assert self.generators_folder is not None, "`generators_folder` is `None`"
        return Path(self.generators_folder)

    def run(self, command: str, stdout=None, cwd=None, ignore_errors=False, env="", quiet=False,
            shell=True, scope="build", stderr=None):
        """ Run a command in the current package context.

        :parameter command: The command to run.
        :parameter stdout: The output stream to write the command output. If ``None``, it defaults to
            the standard output stream.
        :parameter stderr: The error output stream to write the command error output. If ``None``,
            it defaults to the standard error stream.
        :parameter cwd: The current working directory to run the command in.
        :parameter ignore_errors: If ``True``, do not raise an error if the command returns a
            non-zero exit code.
        :parameter env: The environment file to use. If empty, it defaults to ``"conanbuild"`` for
            when ``scope`` is ``build`` or ``"conanrun"`` for ``run``.
            If set to ``None`` explicitly, no environment file will be applied,
            which is useful for commands that do not require any environment.
        :parameter quiet: If ``True``, suppress the output of the command.
        :parameter shell: If ``True``, run the command in a shell. This is passed to the
            underlying ``Popen`` function.
        :parameter scope: The scope of the command, either ``"build"`` or ``"run"``.
        """
        # NOTE: "self.win_bash" is the new parameter "win_bash" for Conan 2.0
        command = self._conan_helpers.cmd_wrapper.wrap(command, conanfile=self)
        if env == "":  # This default allows not breaking for users with ``env=None`` indicating
            # they don't want any env-file applied
            env = "conanbuild" if scope == "build" else "conanrun"

        env = [env] if env and isinstance(env, str) else (env or [])
        assert isinstance(env, list), "env argument to ConanFile.run() should be a list"
        envfiles_folder = self.generators_folder or os.getcwd()
        wrapped_cmd = command_env_wrapper(self, command, env, envfiles_folder=envfiles_folder)
        from conan.internal.util.runners import conan_run
        if not quiet:
            ConanOutput().info(f"{self.display_name}: RUN: {command}", fg=Color.BRIGHT_BLUE)
        ConanOutput().debug(f"{self.display_name}: Full command: {wrapped_cmd}")
        if quiet or ConanOutput.get_output_level() == LEVEL_QUIET:
            stdout = subprocess.DEVNULL if stdout is None else stdout
            stderr = subprocess.DEVNULL if stderr is None else stderr
        retcode = conan_run(wrapped_cmd, cwd=cwd, stdout=stdout, stderr=stderr, shell=shell)
        if not quiet:
            ConanOutput().writeln("")

        if not ignore_errors and retcode != 0:
            raise ConanException("Error %d while executing" % retcode)

        return retcode

    def __repr__(self):
        return self.display_name

    def set_deploy_folder(self, deploy_folder):
        self.cpp_info.deploy_base_folder(self.package_folder, deploy_folder)
        self.buildenv_info.deploy_base_folder(self.package_folder, deploy_folder)
        self.runenv_info.deploy_base_folder(self.package_folder, deploy_folder)
        self.folders.set_base_package(deploy_folder)
