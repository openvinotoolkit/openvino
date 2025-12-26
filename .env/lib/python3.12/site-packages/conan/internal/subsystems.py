"""
Potential scenarios:

- Running from a Windows native "cmd"
  - Targeting Windows native (os.subsystem = None)
    - No need of bash (no conf at all)
    - Need to build in bash (tools.microsoft.bash:subsystem=xxx,
                             tools.microsoft.bash:path=<path>,
                             conanfile.win_bash)
    - Need to run (tests) in bash (tools.microsoft.bash:subsystem=xxx,
                                   tools.microsoft.bash:path=<path>,
                                   conanfile.win_bash_run)
  - Targeting Subsystem (os.subsystem = msys2/cygwin)
    - Always builds and runs in bash (tools.microsoft.bash:path)

- Running from a subsytem terminal (tools.microsoft.bash:subsystem=xxx,
                                    tools.microsoft.bash:path=None) NO ERROR mode for not specifying it? =CURRENT?
  - Targeting Windows native (os.subsystem = None)
  - Targeting Subsystem (os.subsystem = msys2/cygwin)

"""
import os
import platform
import re

from conan.tools.build import cmd_args_to_string
from conan.errors import ConanException

WINDOWS = "windows"
MSYS2 = 'msys2'
MSYS = 'msys'
CYGWIN = 'cygwin'
WSL = 'wsl'  # Windows Subsystem for Linux


def command_env_wrapper(conanfile, command, envfiles, envfiles_folder, scope="build"):
    from conan.tools.env.environment import environment_wrap_command
    if getattr(conanfile, "conf", None) is None:
        # TODO: No conf, no profile defined!! This happens at ``export()`` time
        #  Is it possible to run a self.run() in export() in bash?
        #  Is it necessary? Shouldn't be
        return command

    active = conanfile.conf.get("tools.microsoft.bash:active", check_type=bool)
    subsystem = conanfile.conf.get("tools.microsoft.bash:subsystem")
    if platform.system() == "Windows" and (
            (conanfile.win_bash and scope == "build") or
            (conanfile.win_bash_run and scope == "run")):
        if subsystem is None:
            raise ConanException("win_bash/win_bash_run defined but no "
                                 "tools.microsoft.bash:subsystem")
        if active:
            wrapped_cmd = environment_wrap_command(conanfile, envfiles, envfiles_folder, command)
        else:
            wrapped_cmd = _windows_bash_wrapper(conanfile, command, envfiles, envfiles_folder)
    else:
        wrapped_cmd = environment_wrap_command(conanfile, envfiles, envfiles_folder, command)
    return wrapped_cmd


def _windows_bash_wrapper(conanfile, command, env, envfiles_folder):
    from conan.tools.env import Environment
    from conan.tools.env.environment import environment_wrap_command
    """ Will wrap a unix command inside a bash terminal It requires to have MSYS2, CYGWIN, or WSL"""

    subsystem = conanfile.conf.get("tools.microsoft.bash:subsystem")
    if not platform.system() == "Windows":
        raise ConanException("Command only for Windows operating system")

    shell_path = conanfile.conf.get("tools.microsoft.bash:path")
    if not shell_path:
        raise ConanException("The config 'tools.microsoft.bash:path' is "
                             "needed to run commands in a Windows subsystem")
    shell_path = shell_path.replace("\\", "/")  # Should work in all terminals
    env = env or []
    if subsystem == MSYS2:
        # Configure MSYS2 to inherith the PATH
        msys2_mode_env = Environment()
        _msystem = {"x86": "MINGW32"}.get(conanfile.settings.get_safe("arch"), "MINGW64")
        # https://www.msys2.org/wiki/Launchers/ dictates that the shell should be launched with
        # - MSYSTEM defined
        # - CHERE_INVOKING is necessary to keep the CWD and not change automatically to the user home
        msys2_mode_env.define("MSYSTEM", _msystem)
        msys2_mode_env.define("MSYS2_PATH_TYPE", "inherit")
        msys2_mode_env.unset("ORIGINAL_PATH")
        # So --login do not change automatically to the user home
        msys2_mode_env.define("CHERE_INVOKING", "1")
        path = os.path.join(conanfile.generators_folder, "msys2_mode.bat")
        # Make sure we save pure .bat files, without sh stuff
        wb, conanfile.win_bash = conanfile.win_bash, None
        msys2_mode_env.vars(conanfile, "build").save_bat(path)
        conanfile.win_bash = wb
        env.append(path)

    wrapped_shell = '"%s"' % shell_path if " " in shell_path else shell_path
    wrapped_shell = environment_wrap_command(conanfile, env, envfiles_folder, wrapped_shell,
                                             accepted_extensions=("bat", "ps1"))

    # Wrapping the inside_command enable to prioritize our environment, otherwise /usr/bin go
    # first and there could be commands that we want to skip
    wrapped_user_cmd = environment_wrap_command(conanfile, env, envfiles_folder, command,
                                                accepted_extensions=("sh", ))
    wrapped_user_cmd = _escape_windows_cmd(wrapped_user_cmd)
    # according to https://www.msys2.org/wiki/Launchers/, it is necessary to use --login shell
    # running without it is discouraged
    final_command = '{} --login -c {}'.format(wrapped_shell, wrapped_user_cmd)
    return final_command


def _escape_windows_cmd(command):
    """ To use in a regular windows cmd.exe
        1. Adds escapes so the argument can be unpacked by CommandLineToArgvW()
        2. Adds escapes for cmd.exe so the argument survives cmd.exe's substitutions.

        Useful to escape commands to be executed in a windows bash (msys2, cygwin etc)
    """
    quoted_arg = cmd_args_to_string([command])
    return "".join(["^%s" % arg if arg in r'()%!^"<>&|' else arg for arg in quoted_arg])


def deduce_subsystem(conanfile, scope):
    """ used by:
    - EnvVars: to decide if using :  ; as path separator, translate paths to subsystem
               and decide to generate a .bat or .sh
    - Autotools: to define the full abs path to the "configure" script
    - GnuDeps: to map all the paths from dependencies
    - Aggregation of envfiles: to map each aggregated path to the subsystem
    - unix_path: util for recipes
    """
    scope = "build" if scope is None else scope  # let's assume build context if scope=None
    if scope.startswith("build"):
        the_os = conanfile.settings_build.get_safe("os")
        if the_os is None:
            raise ConanException("The 'build' profile must have a 'os' declared")
    else:
        the_os = conanfile.settings.get_safe("os")

    if not str(the_os).startswith("Windows"):
        return None

    subsystem = conanfile.conf.get("tools.microsoft.bash:subsystem")
    if not subsystem:
        if conanfile.win_bash:
            raise ConanException("win_bash=True but tools.microsoft.bash:subsystem "
                                 "configuration not defined")
        if conanfile.win_bash_run:
            raise ConanException("win_bash_run=True but tools.microsoft.bash:subsystem "
                                 "configuration not defined")
        return WINDOWS
    active = conanfile.conf.get("tools.microsoft.bash:active", check_type=bool)
    if active:
        return subsystem

    if scope.startswith("build"):  # "run" scope do not follow win_bash
        if conanfile.win_bash:
            return subsystem
    elif scope.startswith("run"):
        if conanfile.win_bash_run:
            return subsystem

    return WINDOWS


def subsystem_path(subsystem, path):
    """"Used to translate windows paths to MSYS unix paths like
    c/users/path/to/file. Not working in a regular console or MinGW!
    """
    if subsystem is None or subsystem == WINDOWS:
        return path

    if os.path.exists(path):
        # if the path doesn't exist (and abs) we cannot guess the casing
        path = get_cased_path(path)

    if path.startswith('\\\\?\\'):
        path = path[4:]
    path = path.replace(":/", ":\\")
    append_prefix = re.match(r'[a-z]:\\', path, re.IGNORECASE)
    pattern = re.compile(r'([a-z]):\\', re.IGNORECASE)
    path = pattern.sub('/\\1/', path).replace('\\', '/')

    if append_prefix:
        if subsystem in (MSYS, MSYS2):
            return path.lower()
        elif subsystem == CYGWIN:
            return '/cygdrive' + path.lower()
        elif subsystem == WSL:
            return '/mnt' + path[0:2].lower() + path[2:]
    else:
        return path if subsystem == WSL else path.lower()
    return None


def get_cased_path(name):
    if platform.system() != "Windows":
        return name
    if not os.path.isabs(name):
        name = os.path.abspath(name)

    result = []
    current = name
    while True:
        parent, child = os.path.split(current)
        if parent == current:
            break

        child_cased = child
        if os.path.exists(parent):
            children = os.listdir(parent)
            for c in children:
                if c.upper() == child.upper():
                    child_cased = c
                    break
        result.append(child_cased)
        current = parent
    drive, _ = os.path.splitdrive(current)
    result.append(drive)
    return os.sep.join(reversed(result))
