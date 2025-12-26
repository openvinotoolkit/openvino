import os
import textwrap

from conan.api.output import ConanOutput
from conan.internal import check_duplicated_generator
from conan.internal.api.detect.detect_vs import vs_installation_path
from conan.errors import ConanException, ConanInvalidConfiguration
from conan.tools.scm import Version
from conan.tools.intel.intel_cc import IntelCC
from conan.internal.util.files import save

CONAN_VCVARS = "conanvcvars"


def msvc_platform_from_arch(arch):
    return {"x86": "Win32",
            "x86_64": "x64",
            "armv7": "ARM",
            "armv8": "ARM64",
            "arm64ec": "ARM64EC"}.get(arch)


def check_min_vs(conanfile, version, raise_invalid=True):
    """
    This is a helper method to allow the migration of 1.X -> 2.0 and VisualStudio -> msvc settings
    without breaking recipes.
    The legacy "Visual Studio" with different toolset is not managed, not worth the complexity.

    :param raise_invalid: ``bool`` Whether to raise or return False if the version check fails
    :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
    :param version: ``str`` Visual Studio or msvc version number.
    """
    compiler = conanfile.settings.get_safe("compiler")
    compiler_version = None
    if compiler == "Visual Studio":
        compiler_version = conanfile.settings.get_safe("compiler.version")
        compiler_version = {"17": "193",
                            "16": "192",
                            "15": "191",
                            "14": "190",
                            "12": "180",
                            "11": "170"}.get(compiler_version)
    elif compiler == "msvc":
        compiler_version = conanfile.settings.get_safe("compiler.version")
        msvc_update = conanfile.conf.get("tools.microsoft:msvc_update")
        compiler_update = msvc_update or conanfile.settings.get_safe("compiler.update")
        if compiler_version and compiler_update is not None:
            compiler_version += ".{}".format(compiler_update)

    if compiler_version and Version(compiler_version) < version:
        if raise_invalid:
            msg = f"This package doesn't work with VS compiler version '{compiler_version}'" \
                  f", it requires at least '{version}'"
            raise ConanInvalidConfiguration(msg)
        else:
            return False
    return True


def msvc_version_to_vs_ide_version(version):
    """
    Gets the Visual Studio IDE version given the ``msvc`` compiler one.

    :param version: ``str`` or ``int`` msvc version
    :return: VS IDE version
    """
    _visuals = {'170': '11',
                '180': '12',
                '190': '14',
                '191': '15',
                '192': '16',
                '193': '17',
                '194': '17',  # Note both 193 and 194 belong to VS 17 2022
                '195': '18'}
    return _visuals[str(version)]


def msvc_version_to_toolset_version(version):
    """
    Gets the Visual Studio IDE toolset version given the ``msvc`` compiler one.

    :param version: ``str`` or ``int`` msvc version
    :return: VS IDE toolset version
    """
    toolsets = {'170': 'v110',
                '180': 'v120',
                '190': 'v140',
                '191': 'v141',
                '192': 'v142',
                "193": 'v143',
                "194": 'v143',
                "195": 'v145'}
    return toolsets.get(str(version))


class VCVars:
    """
    VCVars class generator to generate a ``conanvcvars.bat`` script that activates the correct
    Visual Studio prompt.

    This generator will be automatically called by other generators such as ``CMakeToolchain``
    when considered necessary, for example if building with Visual Studio compiler using the
    CMake ``Ninja`` generator, which needs an active Visual Studio prompt.
    Then, it is not necessary to explicitly instantiate this generator in most cases.
    """

    def __init__(self, conanfile):
        """
        :param conanfile: ``ConanFile object`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile

    def generate(self, scope="build"):
        """
        Creates a ``conanvcvars.bat`` file that calls Visual ``vcvars`` with the necessary
        args to activate the correct Visual Studio prompt matching the Conan settings.

        :param scope: ``str`` activation scope, by default "build". It means it will add a
                      call to this ``conanvcvars.bat`` from the aggregating general
                      ``conanbuild.bat``, which is the script that will be called by default
                      in ``self.run()`` calls and build helpers such as ``cmake.configure()``
                      and ``cmake.build()``.
        """
        check_duplicated_generator(self, self._conanfile)
        conanfile = self._conanfile

        os_ = conanfile.settings.get_safe("os")
        build_os_ = conanfile.settings_build.get_safe("os")

        if (os_ != "Windows" and os_ != "WindowsStore") or build_os_ != "Windows":
            return

        compiler = conanfile.settings.get_safe("compiler")
        if compiler not in ("msvc", "clang"):
            return

        vs_install_path = conanfile.conf.get("tools.microsoft.msbuild:installation_path")
        if vs_install_path == "":  # Empty string means "disable"
            return

        vs_version, vcvars_ver = _vcvars_versions(conanfile)
        if vs_version is None:
            return

        vcvarsarch = _vcvars_arch(conanfile)

        winsdk_version = conanfile.conf.get("tools.microsoft:winsdk_version", check_type=str)
        winsdk_version = winsdk_version or conanfile.settings.get_safe("os.version")
        # The vs_install_path is like
        # C:\Program Files (x86)\Microsoft Visual Studio\2019\Community
        # C:\Program Files (x86)\Microsoft Visual Studio\2017\Community
        # C:\Program Files (x86)\Microsoft Visual Studio 14.0
        vcvars = vcvars_command(vs_version, architecture=vcvarsarch, platform_type=None,
                                winsdk_version=winsdk_version, vcvars_ver=vcvars_ver,
                                vs_install_path=vs_install_path)

        content = textwrap.dedent(f"""\
            @echo off
            set __VSCMD_ARG_NO_LOGO=1
            set VSCMD_SKIP_SENDTELEMETRY=1
            echo conanvcvars.bat: Activating environment Visual Studio {vs_version} - {vcvarsarch} - winsdk_version={winsdk_version} - vcvars_ver={vcvars_ver}
            {vcvars}
            """)
        from conan.tools.env.environment import create_env_script
        conan_vcvars_bat = f"{CONAN_VCVARS}.bat"
        create_env_script(conanfile, content, conan_vcvars_bat, scope)
        _create_deactivate_vcvars_file(conanfile, conan_vcvars_bat)

        try:
            is_ps1 = self._conanfile.conf.get("tools.env.virtualenv:powershell", check_type=bool)
            if is_ps1 is not None:
                ConanOutput().warning(
                    "Boolean values for 'tools.env.virtualenv:powershell' are deprecated. "
                    "Please specify 'powershell.exe' or 'pwsh' instead, appending arguments "
                    "if needed (for example: 'powershell.exe -argument'). "
                    "To unset this configuration, use `tools.env.virtualenv:powershell=!`, "
                    "which matches the previous 'False' behavior.",
                    warn_tag="deprecated"
                )
        except ConanException:
            is_ps1 = self._conanfile.conf.get("tools.env.virtualenv:powershell", check_type=str)
        if is_ps1:
            content_ps1 = textwrap.dedent(rf"""
            if (-not $env:VSCMD_ARG_VCVARS_VER){{
                Push-Location "$PSScriptRoot"
                cmd /c "conanvcvars.bat&set" |
                foreach {{
                  if ($_ -match "=") {{
                    $v = $_.split("=", 2); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
                  }}
                }}
                Pop-Location
                write-host conanvcvars.ps1: Activated environment}}
            """).strip()
            conan_vcvars_ps1 = f"{CONAN_VCVARS}.ps1"
            create_env_script(conanfile, content_ps1, conan_vcvars_ps1, scope)
            _create_deactivate_vcvars_file(conanfile, conan_vcvars_ps1)


def _create_deactivate_vcvars_file(conanfile, filename):
    deactivate_filename = f"deactivate_{filename}"
    message = f"[{deactivate_filename}]: *** vcvars env cannot be deactivated ***\n"
    is_ps1 = filename.endswith(".ps1")
    if is_ps1:
        content = f"Write-Host {message}"
    else:
        content = f"echo {message}"
    path = os.path.join(conanfile.generators_folder, deactivate_filename)
    save(path, content)


def vs_ide_version(conanfile):
    """
    Gets the VS IDE version as string. It'll use the ``compiler.version`` (if exists) and/or the
    ``tools.microsoft.msbuild:vs_version`` if ``compiler`` is ``msvc``.

    :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
    :return: ``str`` Visual IDE version number.
    """
    compiler = conanfile.settings.get_safe("compiler")
    compiler_version = conanfile.settings.get_safe("compiler.version")
    if compiler == "msvc":
        toolset_override = conanfile.conf.get("tools.microsoft.msbuild:vs_version", check_type=str)
        if toolset_override:
            visual_version = toolset_override
        else:
            visual_version = msvc_version_to_vs_ide_version(compiler_version)
    else:
        visual_version = compiler_version
    return visual_version


def msvc_runtime_flag(conanfile):
    """
    Gets the MSVC runtime flag given the ``compiler.runtime`` value from the settings.

    :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
    :return: ``str`` runtime flag.
    """
    settings = conanfile.settings
    runtime = settings.get_safe("compiler.runtime")
    if runtime is not None:
        if runtime == "static":
            runtime = "MT"
        elif runtime == "dynamic":
            runtime = "MD"
        else:
            raise ConanException("compiler.runtime should be 'static' or 'dynamic'")
        runtime_type = settings.get_safe("compiler.runtime_type")
        if runtime_type == "Debug":
            runtime = "{}d".format(runtime)
        return runtime
    return ""


def vcvars_command(version, architecture=None, platform_type=None, winsdk_version=None,
                   vcvars_ver=None, start_dir_cd=True, vs_install_path=None):
    """
    Conan-agnostic construction of vcvars command
    https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line

    :param version: ``str`` Visual Studio version.
    :param architecture: ``str`` Specifies the host and target architecture to use.
    :param platform_type: ``str`` Allows you to specify ``store`` or ``uwp`` as the platform type.
    :param winsdk_version: ``str`` Specifies the version of the Windows SDK to use.
    :param vcvars_ver: ``str`` Specifies the Visual Studio compiler toolset to use.
    :param start_dir_cd: ``bool`` If ``True``, the command will execute
                         ``set "VSCMD_START_DIR=%CD%`` at first.
    :param vs_install_path: ``str`` Visual Studio installation path.
    :return: ``str`` complete _vcvarsall_ command.
    """
    cmd = []
    if start_dir_cd:
        cmd.append('set "VSCMD_START_DIR=%CD%" &&')

    # The "call" is useful in case it is called from another .bat script
    cmd.append('call "%s" ' % _vcvars_path(version, vs_install_path))
    if architecture:
        cmd.append(architecture)
    if platform_type:
        cmd.append(platform_type)
    if winsdk_version:
        cmd.append(winsdk_version)
    if vcvars_ver:
        cmd.append("-vcvars_ver=%s" % vcvars_ver)
    return " ".join(cmd)


def _vcvars_path(version, vs_install_path):
    # TODO: This comes from conans/client/tools/win.py vcvars_command()
    vs_path = vs_install_path or vs_installation_path(version)
    if not vs_path or not os.path.isdir(vs_path):
        raise ConanException(f"VS non-existing installation: Visual Studio {version}. "
                             "If using a non-default toolset from a VS IDE version consider "
                             "specifying it with the 'tools.microsoft.msbuild:vs_version' conf")

    if int(version) > 14:
        vcpath = os.path.join(vs_path, "VC/Auxiliary/Build/vcvarsall.bat")
    else:
        vcpath = os.path.join(vs_path, "VC/vcvarsall.bat")
    vcpath = os.path.normpath(vcpath)
    return vcpath


def _vcvars_versions(conanfile):
    compiler = conanfile.settings.get_safe("compiler")
    msvc_update = conanfile.conf.get("tools.microsoft:msvc_update")
    if compiler == "clang":
        # The vcvars only needed for LLVM/Clang and VS ClangCL, who define runtime
        if not conanfile.settings.get_safe("compiler.runtime"):
            # NMake Makefiles will need vcvars activated, for VS target, defined with runtime
            return None, None
        toolset_version = conanfile.settings.get_safe("compiler.runtime_version")
        vs_version = {"v140": "14",
                      "v141": "15",
                      "v142": "16",
                      "v143": "17",
                      "v144": "17",
                      "v145": "18"}.get(toolset_version)
        if vs_version is None:
            raise ConanException("Visual Studio Runtime version (v140-v145) not defined. Please, "
                                 "add the compiler.runtime_version=[v140-v145] setting to your "
                                 "profile.")
        vcvars_ver = {"v140": "14.0",
                      "v141": "14.1",
                      "v142": "14.2",
                      "v143": "14.3",
                      "v144": "14.4",
                      "v145": "14.5"}.get(toolset_version)
        if vcvars_ver and msvc_update is not None:
            vcvars_ver += f"{msvc_update}"
    else:
        vs_version = vs_ide_version(conanfile)
        if int(vs_version) <= 14:
            vcvars_ver = None
        else:
            compiler_version = str(conanfile.settings.compiler.version)
            compiler_update = msvc_update or conanfile.settings.get_safe("compiler.update", "")
            # The equivalent of compiler 19.26 is toolset 14.26
            vcvars_ver = "14.{}{}".format(compiler_version[-1], compiler_update)
    return vs_version, vcvars_ver


def _vcvars_arch(conanfile):
    """
    Computes the vcvars command line architecture based on conanfile settings (host) and
    settings_build.
    """
    settings_host = conanfile.settings
    settings_build = conanfile.settings_build

    arch_host = str(settings_host.arch)
    arch_build = str(settings_build.arch)

    arch = None
    if arch_build == 'x86_64':
        arch = {'x86': "amd64_x86",
                'x86_64': 'amd64',
                'armv7': 'amd64_arm',
                'armv8': 'amd64_arm64',
                'arm64ec': 'amd64_arm64'}.get(arch_host)
    elif arch_build == 'x86':
        arch = {'x86': 'x86',
                'x86_64': 'x86_amd64',
                'armv7': 'x86_arm',
                'armv8': 'x86_arm64'}.get(arch_host)
    elif arch_build == 'armv8':
        arch = {'x86': 'arm64_x86',
                'x86_64': 'arm64_x64',
                'armv7': 'arm64_arm',
                'armv8': 'arm64'}.get(arch_host)

    if not arch:
        raise ConanException('vcvars unsupported architectures %s-%s' % (arch_build, arch_host))

    return arch


def is_msvc(conanfile, build_context=False):
    """
    Validates if the current compiler is ``msvc``.

    :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
    :param build_context: If True, will use the settings from the build context, not host ones
    :return: ``bool`` True, if the host compiler is ``msvc``, otherwise, False.
    """
    if not build_context:
        settings = conanfile.settings
    else:
        settings = conanfile.settings_build
    return settings.get_safe("compiler") == "msvc"


def is_msvc_static_runtime(conanfile):
    """
    Validates when building with Visual Studio or msvc and MT on runtime.

    :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
    :return: ``bool`` True, if ``msvc + runtime MT``. Otherwise, False.
    """
    return is_msvc(conanfile) and "MT" in msvc_runtime_flag(conanfile)


def msvs_toolset(conanfile):
    """
    Returns the corresponding platform toolset based on the compiler setting.
    In case no toolset is configured in the profile, it will return a toolset based on the
    compiler version, otherwise, it will return the toolset from the profile.
    When there is no compiler version neither toolset configured, it will return None
    It supports msvc, intel-cc and clang compilers. For clang, is assumes the ClangCl toolset,
    as provided by the Visual Studio installer.

    :param conanfile: Conanfile instance to access settings.compiler
    :return: A toolset when compiler.version is valid or compiler.toolset is configured. Otherwise, None.
    """
    settings = conanfile.settings
    compiler = settings.get_safe("compiler")
    compiler_version = settings.get_safe("compiler.version")
    if compiler == "msvc":
        subs_toolset = settings.get_safe("compiler.toolset")
        if subs_toolset:
            return subs_toolset
        return msvc_version_to_toolset_version(compiler_version)
    if compiler == "clang":
        return "ClangCl"
    if compiler == "intel-cc":
        return IntelCC(conanfile).ms_toolset
