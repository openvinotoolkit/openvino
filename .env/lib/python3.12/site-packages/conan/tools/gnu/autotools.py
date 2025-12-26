import os
import re

from conan.tools.build import build_jobs, cmd_args_to_string, load_toolchain_args
from conan.internal.subsystems import subsystem_path, deduce_subsystem
from conan.tools.files import chdir
from conan.tools.microsoft import unix_path


def join_arguments(args):
    return " ".join(filter(None, args))


class Autotools(object):

    def __init__(self, conanfile, namespace=None):
        """
        :param conanfile: The current recipe object. Always use ``self``.
        :param namespace: this argument avoids collisions when you have multiple toolchain calls in
                          the same recipe. By setting this argument, the *conanbuild.conf* file used
                          to pass information to the toolchain will be named as:
                          *<namespace>_conanbuild.conf*. The default value is ``None`` meaning that
                          the name of the generated file is *conanbuild.conf*. This namespace must
                          be also set with the same value in the constructor of the AutotoolsToolchain
                          so that it reads the information from the proper file.
        """
        self._conanfile = conanfile

        toolchain_file_content = load_toolchain_args(self._conanfile.generators_folder,
                                                     namespace=namespace)

        self._configure_args = toolchain_file_content.get("configure_args")
        self._make_args = toolchain_file_content.get("make_args")
        self._autoreconf_args = toolchain_file_content.get("autoreconf_args")

    def configure(self, build_script_folder=None, args=None):
        """
        Call the configure script.

        :param args: List of arguments to use for the ``configure`` call.
        :param build_script_folder: Subfolder where the `configure` script is located. If not specified
                                    conanfile.source_folder is used.
        """
        # http://jingfenghanmax.blogspot.com.es/2010/09/configure-with-host-target-and-build.html
        # https://gcc.gnu.org/onlinedocs/gccint/Configure-Terms.html
        script_folder = os.path.join(self._conanfile.source_folder, build_script_folder) \
            if build_script_folder else self._conanfile.source_folder

        configure_args = []
        configure_args.extend(args or [])

        self._configure_args = "{} {}".format(self._configure_args, cmd_args_to_string(configure_args))

        configure_cmd = "{}/configure".format(script_folder)
        subsystem = deduce_subsystem(self._conanfile, scope="build")
        configure_cmd = subsystem_path(subsystem, configure_cmd)
        cmd = '"{}" {}'.format(configure_cmd, self._configure_args)
        self._conanfile.run(cmd)

    def make(self, target=None, args=None, makefile=None):
        """
        Call the make program.

        :param target: (Optional, Defaulted to ``None``): Choose which target to build. This allows
                       building of e.g., docs, shared libraries or install for some AutoTools
                       projects
        :param args: (Optional, Defaulted to ``None``): List of arguments to use for the
                     ``make`` call.
        :param makefile: (Optional, Defaulted to ``None``): Allow specifying a custom makefile to use instead of default "Makefile"
        """
        make_program = self._conanfile.conf.get("tools.gnu:make_program",
                                                default="mingw32-make" if self._use_win_mingw()
                                                else "make")
        str_args = self._make_args
        str_extra_args = " ".join(args) if args is not None else ""
        jobs = ""
        jobs_already_passed = re.search(r"(^-j\d+)|(\W-j\d+\s*)", join_arguments([str_args, str_extra_args]))
        if not jobs_already_passed and "nmake" not in make_program.lower():
            njobs = build_jobs(self._conanfile)
            if njobs:
                jobs = "-j{}".format(njobs)
        str_makefile = f"--file={makefile}" if makefile else None

        command = join_arguments([make_program, str_makefile, target, str_args, str_extra_args, jobs])
        self._conanfile.run(command)

    def install(self, args=None, target=None, makefile=None):
        """
        This is just an "alias" of ``self.make(target="install")`` or ``self.make(target="install-strip")``

        :param args: (Optional, Defaulted to ``None``): List of arguments to use for the
                     ``make`` call. By default an argument ``DESTDIR=unix_path(self.package_folder)``
                     is added to the call if the passed value is ``None``. See more information about
                     :ref:`tools.microsoft.unix_path() function<conan_tools_microsoft_unix_path>`
        :param target: (Optional, Defaulted to ``None``): Choose which target to install.
        :param makefile: (Optional, Defaulted to ``None``): Allow specifying a custom makefile to use instead of default "Makefile"
        """
        if target is None:
            target = "install"
            do_strip = self._conanfile.conf.get("tools.build:install_strip", check_type=bool)
            if do_strip:
                target += "-strip"
        args = args if args else []
        str_args = " ".join(args)
        if "DESTDIR=" not in str_args:
            args.insert(0, "DESTDIR={}".format(unix_path(self._conanfile, self._conanfile.package_folder)))
        self.make(target=target, args=args, makefile=makefile)

    def autoreconf(self, build_script_folder=None, args=None):
        """
        Call ``autoreconf``

        :param args: (Optional, Defaulted to ``None``): List of arguments to use for the
                     ``autoreconf`` call.
        :param build_script_folder: Subfolder where the `configure` script is located. If not specified
                                    conanfile.source_folder is used.
        """
        script_folder = os.path.join(self._conanfile.source_folder, build_script_folder) \
            if build_script_folder else self._conanfile.source_folder
        args = args or []
        command = join_arguments(["autoreconf", self._autoreconf_args, cmd_args_to_string(args)])
        with chdir(self, script_folder):
            self._conanfile.run(command)

    def _use_win_mingw(self):
        os_build = self._conanfile.settings_build.get_safe('os')

        if os_build == "Windows":
            compiler = self._conanfile.settings.get_safe("compiler")
            sub = self._conanfile.settings.get_safe("os.subsystem")
            if sub in ("cygwin", "msys2", "msys") or compiler == "qcc":
                return False
            else:
                if self._conanfile.win_bash:
                    return False
                return True
        return False
