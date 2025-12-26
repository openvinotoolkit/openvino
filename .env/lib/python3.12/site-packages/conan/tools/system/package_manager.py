import platform

from conan.tools.build import cross_building
from conan.internal.graph.graph import CONTEXT_BUILD
from conan.errors import ConanException


class _SystemPackageManagerTool(object):
    mode_check = "check"  # Check if installed, fail if not
    mode_install = "install"
    mode_report = "report"  # Only report what would be installed, no check (can run in any system)
    mode_report_installed = "report-installed"  # report installed and missing packages
    tool_name = None
    version_separator = ""
    install_command = ""
    update_command = ""
    check_command = ""
    check_version_command = ""
    full_package_name = "{name}{arch_separator}{arch_name}"
    accepted_install_codes = [0]
    accepted_update_codes = [0]
    accepted_check_codes = [0, 1]

    def __init__(self, conanfile):
        """
        :param conanfile: The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile
        self._active_tool = self._conanfile.conf.get("tools.system.package_manager:tool", default=self.get_default_tool())
        self._sudo = self._conanfile.conf.get("tools.system.package_manager:sudo", default=False, check_type=bool)
        self._sudo_askpass = self._conanfile.conf.get("tools.system.package_manager:sudo_askpass", default=False, check_type=bool)
        self._mode = self._conanfile.conf.get("tools.system.package_manager:mode", default=self.mode_check)
        self._arch = self._conanfile.settings_build.get_safe('arch') \
            if self._conanfile.context == CONTEXT_BUILD else self._conanfile.settings.get_safe('arch')
        self._arch_names = {}
        self._arch_separator = ""

    def get_default_tool(self):
        os_name = platform.system()
        if os_name in ["Linux", "FreeBSD"]:
            import distro
            os_name = distro.id() or os_name
        elif os_name == "Windows" and self._conanfile.settings.get_safe("os.subsystem") == "msys2":
            os_name = "msys2"
        manager_mapping = {"apt-get": ["Linux", "ubuntu", "debian", "raspbian", "linuxmint", 'astra', 'elbrus', 'altlinux', 'pop'],
                           "apk": ["alpine"],
                           "yum": ["pidora", "scientific", "xenserver", "amazon", "oracle", "amzn",
                                   "almalinux", "rocky"],
                           "dnf": ["fedora", "rhel", "centos", "mageia", "nobara"],
                           "brew": ["Darwin"],
                           "pacman": ["arch", "manjaro", "msys2", "endeavouros"],
                           "choco": ["Windows"],
                           "zypper": ["opensuse", "sles"],
                           "pkg": ["freebsd"],
                           "pkgutil": ["Solaris"]}
        # first check exact match of name
        for tool, distros in manager_mapping.items():
            if os_name in distros:
                return tool
        # in case we did not detect any exact match, check
        # if the name is contained inside the returned distro name
        # like for opensuse, that can have opensuse-version names
        for tool, distros in manager_mapping.items():
            for d in distros:
                if d in os_name:
                    return tool

        # No default package manager was found for the system,
        # so notify the user
        self._conanfile.output.info("A default system package manager couldn't be found for {}, "
                                    "system packages will not be installed.".format(os_name))

    def _split_package_name(self, package, host_package):

        name, version = (package.split("=")[0], package.split("=")[1]) if "=" in package else (package, "")
        arch_separator, arch_name = "", ""
        version_separator = self.version_separator if version else ""

        if self._arch in self._arch_names and cross_building(self._conanfile) and host_package:
            arch_separator = self._arch_separator
            arch_name = self._arch_names.get(self._arch)
        return name, version, arch_separator, arch_name, version_separator

    def get_package_name(self, package, host_package=True):
        # Only if the package is for building, for example a library,
        # we should add the host arch when cross building.
        # If the package is a tool that should be installed on the current build
        # machine we should not add the arch.

        name, version, arch_separator, arch_name, version_separator = self._split_package_name(package, host_package)

        return self.full_package_name.format(name=name,
                                             arch_separator=arch_separator,
                                             arch_name=arch_name,
                                             version_separator=version_separator,
                                             version=version)

    @property
    def sudo_str(self):
        sudo = "sudo " if self._sudo else ""
        askpass = "-A " if self._sudo and self._sudo_askpass else ""
        return "{}{}".format(sudo, askpass)

    def run(self, method, *args, **kwargs):
        if self._active_tool == self.__class__.tool_name:
            return method(*args, **kwargs)

    def _conanfile_run(self, command, accepted_returns, quiet=True):
        # When checking multiple packages, this is too noisy
        ret = self._conanfile.run(command, ignore_errors=True, quiet=quiet)
        if ret not in accepted_returns:
            raise ConanException("Command '%s' failed" % command)
        return ret

    def install_substitutes(self, *args, **kwargs):
        """
        Will try to call the install() method with  several lists of packages passed as a
        variable number of parameters. This is useful if, for example, the names of the
        packages are different from one distro or distro version to another. For example,
        ``libxcb`` for ``Apt`` is named ``libxcb-util-dev`` in Ubuntu >= 15.0 and ``libxcb-util0-dev``
        for other versions. You can call to:

           .. code-block:: python

            # will install the first list of packages that succeeds in the installation
            Apt.install_substitutes(["libxcb-util-dev"], ["libxcb-util0-dev"])

        :param packages_alternatives: try to install the list of packages passed as a parameter.
        :param update: try to update the package manager database before checking and installing.
        :param check: check if the packages are already installed before installing them.
        :return: the return code of the executed package manager command.
        """
        return self.run(self._install_substitutes, *args, **kwargs)

    def install(self, *args, **kwargs):
        """
        Will try to install the list of packages passed as a parameter. Its
        behaviour is affected by the value of ``tools.system.package_manager:mode``
        :ref:`configuration<conan_tools_system_package_manager_config>`.

        :param packages: try to install the list of packages passed as a parameter.
        :param update: try to update the package manager database before checking and installing.
        :param check: check if the packages are already installed before installing them.
        :return: the return code of the executed package manager command.
        """
        return self.run(self._install, *args, **kwargs)

    def update(self, *args, **kwargs):
        """
        Update the system package manager database. Its behaviour is affected by
        the value of ``tools.system.package_manager:mode``
        :ref:`configuration<conan_tools_system_package_manager_config>`.

        :return: the return code of the executed package manager update command.
        """
        return self.run(self._update, *args, **kwargs)

    def check(self, *args, **kwargs):
        """
        Check if the list of packages passed as parameter are already installed.

        :param packages: list of packages to check.
        :return: list of packages from the packages argument that are not installed in the system.
        """
        return self.run(self._check, *args, **kwargs)

    def _install_substitutes(self, *packages_substitutes, update=False, check=True, **kwargs):
        errors = []
        for packages in packages_substitutes:
            try:
                return self.install(packages, update, check, **kwargs)
            except ConanException as e:
                errors.append(e)

        for error in errors:
            self._conanfile.output.warning(str(error))
        raise ConanException("None of the installs for the package substitutes succeeded.")

    def _install(self, packages, update=False, check=True, host_package=True, **kwargs):
        pkgs = self._conanfile.system_requires.setdefault(self._active_tool, {})
        install_pkgs = pkgs.setdefault("install", [])
        install_pkgs.extend(p for p in packages if p not in install_pkgs)
        if self._mode == self.mode_report:
            return

        if check or self._mode in (self.mode_check, self.mode_report_installed):
            packages = self.check(packages, host_package=host_package)
            missing_pkgs = pkgs.setdefault("missing", [])
            missing_pkgs.extend(p for p in packages if p not in missing_pkgs)
        if self._mode == self.mode_report_installed:
            return

        if self._mode == self.mode_check and packages:
            raise ConanException("System requirements: '{0}' are missing but can't install "
                                 "because tools.system.package_manager:mode is '{1}'."
                                 "Please update packages manually or set "
                                 "'tools.system.package_manager:mode' "
                                 "to '{2}' in the [conf] section of the profile, "
                                 "or in the command line using "
                                 "'-c tools.system.package_manager:mode={2}'".format(", ".join(packages),
                                                                                     self.mode_check,
                                                                                     self.mode_install))
        elif packages:
            if update:
                self.update()
            packages_arch = [self.get_package_name(package, host_package=host_package) for package in packages]
            if packages_arch:
                command = self.install_command.format(sudo=self.sudo_str,
                                                      tool=self.tool_name,
                                                      packages=" ".join(packages_arch),
                                                      **kwargs)
                return self._conanfile_run(command, self.accepted_install_codes, quiet=False)
        else:
            self._conanfile.output.info("System requirements: {} already "
                                        "installed".format(" ".join(packages)))

    def _update(self):
        # we just update the package manager database in case we are in 'install mode'
        # in case we are in check mode just ignore
        if self._mode == self.mode_install:
            command = self.update_command.format(sudo=self.sudo_str, tool=self.tool_name)
            return self._conanfile_run(command, self.accepted_update_codes, quiet=False)

    def _check(self, packages, host_package=True):
        missing = [pkg for pkg in packages if self.check_package(pkg, host_package) != 0]
        return missing

    def check_package(self, package, host_package=True):
        name, version, arch_separator, arch_name, _ = self._split_package_name(package, host_package)

        check_arch = self._arch if host_package else self._conanfile.settings_build.get_safe('arch')
        arch_package = arch_name or self._arch_names.get(check_arch)
        package = self.full_package_name.format(name=name,
                                                arch_separator=arch_separator,
                                                arch_name=arch_name,
                                                version="",
                                                version_separator="")
        command = self.check_command.format(tool=self.tool_name, package=package, arch_package=arch_package, base_name=name)

        if version:
            if self.check_version_command:
                command = self.check_version_command.format(tool=self.tool_name, package=package, version=version, arch_package=arch_package,
                                                            base_name=name)
            else:
                self._conanfile.output.warning(f"System requirements: \"{self.tool_name}\" doesn't support package versions,"
                                               f" \"{package}\" will be installed without a specific version.")
        return self._conanfile_run(command, self.accepted_check_codes)


class Apt(_SystemPackageManagerTool):
    # TODO: apt? apt-get?
    tool_name = "apt-get"
    version_separator = "="
    full_package_name = "{name}{arch_separator}{arch_name}{version_separator}{version}"
    install_command = "{sudo}{tool} install -y {recommends}{packages}"
    update_command = "{sudo}{tool} update"
    check_command = r"dpkg-query -W -f='${{Architecture}}\n' {base_name} | grep -qEx '({arch_package}|all)'"
    check_version_command = r"dpkg-query -W -f='${{Architecture}} ${{Version}}\n' {base_name} | grep -qEx '({arch_package}|all) {version}'"

    def __init__(self, conanfile, arch_names=None):
        """
        :param conanfile: The current recipe object. Always use ``self``.
        :param arch_names: This argument maps the Conan architecture setting with the package manager
                           tool architecture names. It is ``None`` by default, which means that it will use a
                           default mapping for the most common architectures. For example, if you are using
                           ``x86_64`` Conan architecture setting, it will map this value to ``amd64`` for *Apt* and
                           try to install the ``<package_name>:amd64`` package.
        """
        super(Apt, self).__init__(conanfile)
        self._arch_names = {"x86_64": "amd64",
                            "x86": "i386",
                            "ppc32": "powerpc",
                            "ppc64le": "ppc64el",
                            "armv7": "arm",
                            "armv7hf": "armhf",
                            "armv8": "arm64",
                            "s390x": "s390x"} if arch_names is None else arch_names

        self._arch_separator = ":"

    def install(self, packages, update=False, check=True, recommends=False, host_package=True):
        """
        Will try to install the list of packages passed as a parameter. Its
        behaviour is affected by the value of ``tools.system.package_manager:mode``
        :ref:`configuration<conan_tools_system_package_manager_config>`.

        :param packages: try to install the list of packages passed as a parameter.
        :param update: try to update the package manager database before checking and installing.
        :param check: check if the packages are already installed before installing them.
        :param host_package: install the packages for the host machine architecture (the machine
               that will run the software), it has an effect when cross building.
        :param recommends: if the parameter ``recommends`` is ``False`` it will add the
               ``'--no-install-recommends'`` argument to the *apt-get* command call.
        :return: the return code of the executed apt command.
        """
        recommends_str = '' if recommends else '--no-install-recommends '
        return super(Apt, self).install(packages, update=update, check=check,
                                        host_package=host_package, recommends=recommends_str)


class Yum(_SystemPackageManagerTool):
    tool_name = "yum"
    version_separator = "-"
    full_package_name = "{name}{version_separator}{version}{arch_separator}{arch_name}"
    install_command = "{sudo}{tool} install -y {packages}"
    update_command = "{sudo}{tool} check-update -y"
    check_command = "rpm -q {package}"
    check_version_command = "rpm -q {package}-{version}"
    accepted_update_codes = [0, 100]

    def __init__(self, conanfile, arch_names=None):
        """
        :param conanfile: the current recipe object. Always use ``self``.
        :param arch_names: this argument maps the Conan architecture setting with the package manager
                           tool architecture names. It is ``None`` by default, which means that it will use a
                           default mapping for the most common architectures. For example, if you are using
                           ``x86`` Conan architecture setting, it will map this value to ``i?86`` for *Yum* and
                           try to install the ``<package_name>.i?86`` package.
        """
        super(Yum, self).__init__(conanfile)
        self._arch_names = {"x86_64": "x86_64",
                            "x86": "i?86",
                            "ppc32": "powerpc",
                            "ppc64le": "ppc64le",
                            "armv7": "armv7",
                            "armv7hf": "armv7hl",
                            "armv8": "aarch64",
                            "s390x": "s390x"} if arch_names is None else arch_names
        self._arch_separator = "."


class Dnf(Yum):
    tool_name = "dnf"
    version_separator = "-"
    full_package_name = "{name}{version_separator}{version}{arch_separator}{arch_name}"
    check_version_command = "rpm -q {package}-{version}"


class Brew(_SystemPackageManagerTool):
    tool_name = "brew"
    version_separator = "@"
    full_package_name = "{name}{version_separator}{version}"
    install_command = "{sudo}{tool} install {packages}"
    update_command = "{sudo}{tool} update"
    check_command = 'test -n "$({tool} ls --versions {package})"'
    check_version_command = 'brew list --versions {package} | grep "{version}"'


class Pkg(_SystemPackageManagerTool):
    tool_name = "pkg"
    version_separator = "-"
    full_package_name = "{name}{version_separator}{version}"
    install_command = "{sudo}{tool} install -y {packages}"
    update_command = "{sudo}{tool} update"
    check_command = "{tool} info {package}"
    check_version_command = "{tool} info {package} | grep \"Version: {version}\""


class PkgUtil(_SystemPackageManagerTool):
    tool_name = "pkgutil"
    version_separator = "@"
    full_package_name = "{name}{version_separator}{version}"
    install_command = "{sudo}{tool} --install --yes {packages}"
    update_command = "{sudo}{tool} --catalog"
    check_command = 'test -n "`{tool} --list {package}`"'


class Chocolatey(_SystemPackageManagerTool):
    tool_name = "choco"
    version_separator = " --version "
    full_package_name = "{name}{version_separator}{version}"
    install_command = "{tool} install --yes {packages}"
    update_command = "{tool} outdated"
    check_command = '{tool} list --exact {package} | findstr /c:"1 packages installed."'
    check_version_command = '{tool} list --local-only {package} | findstr /i "{version}"'


class PacMan(_SystemPackageManagerTool):
    tool_name = "pacman"
    install_command = "{sudo}{tool} -S --noconfirm {packages}"
    update_command = "{sudo}{tool} -Syyu --noconfirm"
    check_command = "{tool} -Qi {package}"

    def __init__(self, conanfile, arch_names=None):
        """
        :param conanfile: the current recipe object. Always use ``self``.
        :param arch_names: this argument maps the Conan architecture setting with the package manager
                           tool architecture names. It is ``None`` by default, which means that it will use a
                           default mapping for the most common architectures. If you are using
                           ``x86`` Conan architecture setting, it will map this value to ``lib32`` for *PacMan* and
                           try to install the ``<package_name>-lib32`` package.
        """
        super(PacMan, self).__init__(conanfile)
        self._arch_names = {"x86": "lib32"} if arch_names is None else arch_names
        self._arch_separator = "-"


class Apk(_SystemPackageManagerTool):
    tool_name = "apk"
    version_separator = "="
    full_package_name = "{name}{version_separator}{version}"
    install_command = "{sudo}{tool} add --no-cache {packages}"
    update_command = "{sudo}{tool} update"
    check_command = "{tool} info -e {package}"
    check_version_command = "{tool} info {package} | grep \"{version}\""

    def __init__(self, conanfile, _arch_names=None):
        """
        Constructor method.
        Note that *Apk* does not support architecture names since Alpine Linux does not support
        multiarch. Therefore, the ``arch_names`` argument is ignored.

        :param conanfile: the current recipe object. Always use ``self``.
        """
        super(Apk, self).__init__(conanfile)
        self._arch_names = {}
        self._arch_separator = ""


class Zypper(_SystemPackageManagerTool):
    tool_name = "zypper"
    version_separator = "="  # < or >
    full_package_name = "{name}{arch_separator}{arch_name}{version_separator}{version}"
    install_command = "{sudo}{tool} --non-interactive in {packages}"
    update_command = "{sudo}{tool} --non-interactive ref"
    check_command = "rpm -q {package}"
    check_version_command = "rpm -q {package}-{version}"
