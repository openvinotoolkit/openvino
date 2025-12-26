from pathlib import PurePosixPath, PureWindowsPath, Path
from conan.api.output import Color, ConanOutput
from conan.internal.util.runners import conan_run
from conan.internal.subsystems import subsystem_path
from conan.tools.files import save
from io import StringIO
import tempfile
import os

def wsl_info(msg, error=False):
    fg=Color.BRIGHT_MAGENTA
    if error:
        fg=Color.BRIGHT_RED
    ConanOutput().status('\n┌'+'─'*(2+len(msg))+'┐', fg=fg)
    ConanOutput().status(f'| {msg} |', fg=fg)
    ConanOutput().status('└'+'─'*(2+len(msg))+'┘\n', fg=fg)


class WSLRunner:
    def __init__(self, conan_api, command, host_profile, build_profile, args, raw_args):
        self.conan_api = conan_api
        self.command = command
        self.host_profile = host_profile
        self.build_profile = build_profile
        self.remote_host_profile = None
        self.remote_build_profile = None
        self.remote_python_command = None
        self.remote_conan = None
        self.remote_conan_home = None
        self.args = args
        self.raw_args = raw_args

        # to pass to wsl.exe (optional, otherwise run with defaults)
        distro = host_profile.runner.get("distribution", None)
        user = host_profile.runner.get("user", None)

        self.shared_cache = host_profile.runner.get("shared_cache", False)
        if self.shared_cache:
            storage_path = Path(conan_api.config.home()) / 'p' # TODO: there's an API for this!!
            self.remote_conan_cache = subsystem_path("wsl", storage_path.as_posix())

    def run(self):
        self.ensure_runner_environment()

        raw_args = self.raw_args
        current_path = Path(self.args.path).resolve()
        current_path_wsl = subsystem_path("wsl", current_path.as_posix())

        raw_args[raw_args.index(self.args.path)] = current_path_wsl
        raw_args = " ".join(raw_args)

        with tempfile.TemporaryDirectory() as tmp_dir:
            if not self.shared_cache:
                create_json = PureWindowsPath(tmp_dir).joinpath("create.json").as_posix()
                raw_args += f" --format=json > {create_json}"
            tmp_dir_wsl = subsystem_path("wsl", tmp_dir)
            command = f"wsl.exe --cd {tmp_dir_wsl} -- CONAN_RUNNER_ENVIRONMENT=1 CONAN_HOME={self.remote_conan_home} {self.remote_conan} create {raw_args}"
            rc = conan_run(command)
            if rc == 0 and not self.shared_cache:
                create_json_wsl = subsystem_path("wsl", create_json)
                pkglist_json = PureWindowsPath(tmp_dir).joinpath("pkglist.json").as_posix()
                pkglist_json_wsl = subsystem_path("wsl", pkglist_json)

                saved_cache = PureWindowsPath(tmp_dir).joinpath("saved_cache.tgz").as_posix()
                saved_cache_wsl = subsystem_path("wsl", saved_cache)
                conan_run(f"wsl.exe --cd {tmp_dir_wsl} -- CONAN_RUNNER_ENVIRONMENT=1 CONAN_HOME={self.remote_conan_home} {self.remote_conan} list --graph={create_json_wsl} --format=json > {pkglist_json}")
                conan_run(f"wsl.exe --cd {tmp_dir_wsl} -- CONAN_RUNNER_ENVIRONMENT=1 CONAN_HOME={self.remote_conan_home} {self.remote_conan} cache save --list={pkglist_json_wsl} --file {saved_cache_wsl}")
                self.conan_api.cache.restore(saved_cache)
            else:
                pass
        #print(command)

    def ensure_runner_environment(self):
        stdout = StringIO()
        stderr = StringIO()

        ret = conan_run('wsl.exe echo $HOME', stdout=stdout)
        if ret == 0:
            remote_home = PurePosixPath(stdout.getvalue().strip())
            stdout = StringIO()

        remote_conan = remote_home / ".conan2remote" / "venv" / "bin" / "conan"
        self.remote_conan = remote_conan.as_posix()

        wsl_info(self.remote_conan)

        conan_home = remote_home / ".conan2remote" / "conan_home"
        self.remote_conan_home = conan_home

        has_conan = conan_run(f"wsl.exe CONAN_HOME={conan_home.as_posix()} {remote_conan} --version", stdout=stdout, stderr=stderr) == 0

        if not has_conan:
            wsl_info("Bootstrapping Conan in remote")
            conan_run(f"wsl.exe mkdir -p {remote_home}/.conan2remote")
            venv = remote_home / ".conan2remote"/ "venv"
            python = venv / "bin" / "python"
            self.remote_python_command = python
            conan_run(f"wsl.exe python3 -m venv {venv.as_posix()}")
            conan_run(f"wsl.exe {python} -m pip install pip wheel --upgrade")
            conan_run(f"wsl.exe {python} -m pip install git+https://github.com/conan-io/conan@feature/docker_wrapper")
            conan_run(f"wsl.exe CONAN_HOME={conan_home.as_posix()} {remote_conan} --version", stdout=stdout)

        remote_conan_version = stdout.getvalue().strip()
        wsl_info(f"Remote conan version: {remote_conan_version}")
        stdout = StringIO()
        stderr = StringIO()

        # If this command succeeds, great - if not because it already exists, ignore
        conan_run(f"wsl.exe CONAN_HOME={conan_home.as_posix()} {remote_conan} profile detect", stdout=stdout, stderr=stderr)


        conf_content = f"core.cache:storage_path={self.remote_conan_cache}\n"  if self.shared_cache else ""
        with tempfile.TemporaryDirectory() as tmp:
            global_conf = os.path.join(tmp, "global.conf")
            save(None, path=global_conf, content=conf_content)
            global_conf_wsl = subsystem_path("wsl", global_conf)
            remote_global_conf = self.remote_conan_home.joinpath("global.conf")
            conan_run(f"wsl.exe cp {global_conf_wsl} {remote_global_conf}")

        self._copy_profiles()

    def _copy_profiles(self):
        # TODO: questionable choices, may fail

        # Note: see the use of \\wsl$\<DistroName>\, we could place the files
        #       directly. We would need to work out the exact distro name first
        profiles = {
            self.args.profile_host[0]: self.host_profile.dumps(),
            self.args.profile_build[0]: self.build_profile.dumps()
        }

        with tempfile.TemporaryDirectory() as tmp:
            # path = os.path.join(tmp, 'something')
            for name, contents in profiles.items():
                outfile = os.path.join(tmp, name)
                save(None, path=outfile, content=contents)
                outfile_wsl = subsystem_path("wsl", outfile)
                remote_profile = self.remote_conan_home.joinpath("profiles").as_posix() + "/"

                # This works but copies the file with executable attribute
                conan_run(f"wsl.exe cp {outfile_wsl} {remote_profile}")


