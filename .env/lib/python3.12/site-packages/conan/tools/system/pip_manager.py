import platform
import os
import shutil

from conan.tools.build import cmd_args_to_string
from conan.tools.env.environment import Environment
from conan.errors import ConanException


class PipEnv:

    def __init__(self, conanfile, folder=None, name=""):
        """
        :param conanfile: The current conanfile "self"
        :param folder: Optional folder, by default the "build_folder"
        :param name: Optional name for the virtualenv, by default "conan_pipenv"
        """
        self._conanfile = conanfile
        self.env_name = f"conan_pipenv{f'_{name}' if name else ''}"
        self._env_dir = os.path.abspath(os.path.join(folder or conanfile.build_folder,
                                                     self.env_name))
        bins = "Scripts" if platform.system() == "Windows" else "bin"
        self.bin_dir = os.path.join(self._env_dir, bins)
        pyexe = "python.exe" if platform.system() == "Windows" else "python"
        self._python_exe = os.path.join(self.bin_dir, pyexe)
        self._create_venv()

    def generate(self):
        """
        Create a conan environment to use the python venv in the next steps of the conanfile.
        """
        env = Environment()
        env.prepend_path("PATH", self.bin_dir)
        env.vars(self._conanfile).save_script(self.env_name)

    @staticmethod
    def _default_python():
        python = "python" if platform.system() == "Windows" else "python3"
        default_python = shutil.which(python)
        return os.path.realpath(default_python) if default_python else None

    def _create_venv(self):
        python_interpreter = self._conanfile.conf.get("tools.system.pipenv:python_interpreter")
        python_interpreter = python_interpreter or self._default_python()
        if not python_interpreter:
            raise ConanException("PipEnv could not find a Python executable path. Please, install "
                                 "Python system-wide or set the "
                                 "'tools.system.pipenv:python_interpreter' "
                                 "conf to the full path of a Python executable")

        try:
            self._conanfile.run(cmd_args_to_string([python_interpreter, '-m', 'venv',
                                                    self._env_dir]))
        except ConanException as e:
            raise ConanException(f"PipEnv could not create a Python virtual "
                                 f"environment using '{python_interpreter}': {e}")

    def install(self, packages, pip_args=None):
        """
        Will try to install the list of pip packages passed as a parameter.

        :param packages: try to install the list of pip packages passed as a parameter.
        :param pip_args: additional argument list to be passed to the 'pip install' command,
                         e.g.: ['--no-cache-dir', '--index-url', 'https://my.pypi.org/simple'].
                         Defaults to ``None``.
        :return: the return code of the executed pip command.
        """
        args = [self._python_exe, "-m", "pip", "install", "--disable-pip-version-check"]
        if pip_args:
            args += list(pip_args)
        args += list(packages)
        command = cmd_args_to_string(args)
        return self._conanfile.run(command)
