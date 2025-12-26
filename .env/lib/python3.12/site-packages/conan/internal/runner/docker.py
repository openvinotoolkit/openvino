from argparse import Namespace
import os
import sys
import json
import platform
import shutil
from typing import Optional, NamedTuple, Dict, List
import yaml
from conan.api.conan_api import ConanAPI
from conan.api.model import ListPattern
from conan.api.output import Color, ConanOutput
from conan.cli import make_abs_path
from conan.internal.runner import RunnerException
from conan.errors import ConanException
from pathlib import Path
from conan.internal.model.profile import Profile
from conan.internal.model.version import Version
from conan.internal.runner.output import RunnerOutput
from conan.internal.conan_app import ConanApp


class _ContainerConfig(NamedTuple):
    class Build(NamedTuple):
        dockerfile: Optional[str] = None
        build_context: Optional[str] = None
        build_args: Optional[Dict[str, str]] = None
        cache_from: Optional[List[str]] = None
        platform: Optional[str] = None

    class Run(NamedTuple):
        name: Optional[str] = None
        environment: Optional[Dict[str, str]] = None
        user: Optional[str] = None
        privileged: Optional[bool] = None
        cap_add: Optional[List[str]] = None
        security_opt: Optional[List[str]] = None
        volumes: Optional[Dict[str, str]] = None
        network: Optional[str] = None

    image: Optional[str] = None
    build: Build = Build()
    run: Run = Run()

    @staticmethod
    def load(file_path: Optional[str]) -> '_ContainerConfig':
        # Container config
        # https://containers.dev/implementors/json_reference/
        if not file_path:
            return _ContainerConfig()

        def _instans_or_error(value, obj):
            if value and (not isinstance(value, obj)):
                raise ConanException(f"docker runner configfile syntax error: {value} must be a {obj.__name__}")
            return value
        with open(file_path, 'r') as f:
            runnerfile = yaml.safe_load(f)
        return _ContainerConfig(
            image=_instans_or_error(runnerfile.get('image'), str),
            build=_ContainerConfig.Build(
                dockerfile=_instans_or_error(runnerfile.get('build', {}).get('dockerfile'), str),
                build_context=_instans_or_error(runnerfile.get('build', {}).get('build_context'), str),
                build_args=_instans_or_error(runnerfile.get('build', {}).get('build_args'), dict),
                cache_from=_instans_or_error(runnerfile.get('build', {}).get('cacheFrom'), list),
                platform=_instans_or_error(runnerfile.get('build', {}).get('platform'), str),
            ),
            run=_ContainerConfig.Run(
                name=_instans_or_error(runnerfile.get('run', {}).get('name'), str),
                environment=_instans_or_error(runnerfile.get('run', {}).get('containerEnv'), dict),
                user=_instans_or_error(runnerfile.get('run', {}).get('containerUser'), str),
                privileged=_instans_or_error(runnerfile.get('run', {}).get('privileged'), bool),
                cap_add=_instans_or_error(runnerfile.get('run', {}).get('capAdd'), list),
                security_opt=_instans_or_error(runnerfile.get('run', {}).get('securityOpt'), list),
                volumes=_instans_or_error(runnerfile.get('run', {}).get('mounts'), dict),
                network=_instans_or_error(runnerfile.get('run', {}).get('network'), str),
            )
        )


class DockerRunner:
    def __init__(self, conan_api: ConanAPI, command: str, host_profile: Profile, build_profile: Profile, args: Namespace, raw_args: list[str]):
        self.logger = ConanOutput()
        self.docker_client = self._initialize_docker_client()
        self.docker_api = self.docker_client.api
        self.conan_api = conan_api
        self.build_profile = build_profile
        self.abs_host_path = self._get_abs_host_path(args.path)
        self.args = args
        if args.format:
            raise ConanException("format argument is forbidden if running in a docker runner")

        self.configfile = _ContainerConfig.load(host_profile.runner.get('configfile'))
        self.dockerfile = host_profile.runner.get('dockerfile') or self.configfile.build.dockerfile
        self.docker_build_context = host_profile.runner.get('build_context') or self.configfile.build.build_context
        self.platform = host_profile.runner.get('platform') or self.configfile.build.platform
        self.image = host_profile.runner.get('image') or self.configfile.image
        if not (self.dockerfile or self.image):
            raise ConanException("'dockerfile' or docker image name is needed")
        self.image = self.image or 'conan-runner-default'
        self.name = self.configfile.run.name or host_profile.runner.get("name", "conan-runner-docker")
        self.remove = str(host_profile.runner.get('remove', 'false')).lower() == 'true'
        self.cache = str(host_profile.runner.get('cache', 'clean'))
        if self.cache not in ['clean', 'copy', 'shared']:
            raise ConanException(f'Invalid cache value: "{self.cache}". Valid values are: clean, copy, shared')
        self.container = None
        self.raw_args = raw_args
        self.command = command
        self.runner_logger = RunnerOutput(self.name)

    def run(self) -> None:
        """
        Run conan inside a Docker container
        """
        self._build_image()
        self._start_container()
        try:
            self._init_container()
            self._run_command(self.command)
            self._update_local_cache()
        except RunnerException as e:
            raise ConanException(f'"{e.command}" inside docker fail')
        finally:
            if self.container:
                error = sys.exc_info()[0] is not None  # Check if error has been raised
                log = self.logger.error if error else self.logger.status
                log('Stopping container')
                self.container.stop()
                if self.remove:
                    log('Removing container')
                    self.container.remove()

    def _initialize_docker_client(self):
        import docker
        import docker.api.build

        docker_base_urls = [
            None,  # Default docker configuration
            os.environ.get('DOCKER_HOST'),  # From DOCKER_HOST environment variable
            'unix:///var/run/docker.sock',  # Default Linux socket
            f'unix://{os.path.expanduser("~")}/.rd/docker.sock'  # Rancher Linux socket
        ]

        for base_url in docker_base_urls:
            try:
                self.logger.verbose(f'Trying to connect to Docker: "{base_url or "default"}"')
                client = docker.DockerClient(base_url=base_url, version='auto')
                self.logger.verbose(f'Connected to Docker: "{base_url or "default"}"')
                docker.api.build.process_dockerfile = lambda dockerfile, path: ('Dockerfile', dockerfile)
                return client
            except Exception:
                continue
        raise ConanException("Docker Client failed to initialize."
                             "\n - Check if docker is installed and running"
                             "\n - Run 'pip install conan[runners]'")

    def _get_abs_host_path(self, path: str) -> Path:
        abs_path = make_abs_path(path)
        if abs_path is None:
            raise ConanException("Could not determine the absolute path.")
        return Path(abs_path)

    def _build_image(self) -> None:
        if not self.dockerfile:
            return
        self.logger.status(f'Building the Docker image: {self.image}')
        dockerfile_file_path = self.dockerfile
        if os.path.isdir(self.dockerfile):
            dockerfile_file_path = os.path.join(self.dockerfile, 'Dockerfile')
        with open(dockerfile_file_path) as f:
            build_path = self.docker_build_context or os.path.dirname(dockerfile_file_path)
            self.logger.highlight(f"Dockerfile path: '{dockerfile_file_path}'")
            self.logger.highlight(f"Docker build context: '{build_path}'\n")
            docker_build_logs = self.docker_api.build(
                path=build_path,
                dockerfile=f.read(),
                tag=self.image,
                platform=self.platform,
                buildargs=self.configfile.build.build_args,
                cache_from=self.configfile.build.cache_from,
            )
        for chunk in docker_build_logs:
            for line in chunk.decode("utf-8").split('\r\n'):
                if line:
                    stream = json.loads(line).get('stream')
                    if stream:
                        self.logger.status(stream.strip())

    def _start_container(self) -> None:
        volumes, environment = self._create_runner_environment()
        try:
            if self.docker_client.containers.list(all=True, filters={'name': self.name}):
                self.logger.status('Starting the docker container', fg=Color.BRIGHT_MAGENTA)
                self.container = self.docker_client.containers.get(self.name)
                self.container.start()
            else:
                if self.configfile.run.environment:
                    environment.update(self.configfile.run.environment)
                if self.configfile.run.volumes:
                    volumes.update(self.configfile.run.volumes)
                self.logger.status('Creating the docker container', fg=Color.BRIGHT_MAGENTA)
                self.container = self.docker_client.containers.run(
                    self.image,
                    "/bin/bash -c 'while true; do sleep 30; done;'",
                    name=self.name,
                    volumes=volumes,
                    environment=environment,
                    user=self.configfile.run.user,
                    privileged=self.configfile.run.privileged,
                    cap_add=self.configfile.run.cap_add,
                    security_opt=self.configfile.run.security_opt,
                    detach=True,
                    auto_remove=False,
                    network=self.configfile.run.network)
            self.logger.status(f'Container {self.name} running', fg=Color.BRIGHT_MAGENTA)
        except Exception as e:
            raise ConanException(f'Imposible to run the container "{self.name}" with image "{self.image}"'
                                 f'\n\n{str(e)}')

    def _run_command(self, command: str, workdir: Optional[str] = None, verbose: bool = True) -> tuple[str, str]:
        workdir = workdir or self.abs_docker_path
        log = self.runner_logger.status if verbose else self.runner_logger.verbose
        log(f'$ {command}', fg=Color.BLUE)
        exec_instance = self.docker_api.exec_create(self.container.id, f"/bin/bash -c '{command}'", workdir=workdir, tty=True)
        exec_output = self.docker_api.exec_start(exec_instance['Id'], tty=True, stream=True, demux=True,)
        stderr_log, stdout_log = '', ''
        try:
            for (stdout_out, stderr_out) in exec_output:
                if stdout_out is not None:
                    decoded = stdout_out.decode('utf-8', errors='ignore').strip()
                    stdout_log += decoded
                    log(decoded)
                if stderr_out is not None:
                    decoded = stderr_out.decode('utf-8', errors='ignore').strip()
                    stderr_log += decoded
                    log(decoded)
        except Exception as e:
            if platform.system() == 'Windows':
                import pywintypes
                if isinstance(e, pywintypes.error):
                    pass
            else:
                raise e
        exit_metadata = self.docker_api.exec_inspect(exec_instance['Id'])
        if exit_metadata['Running'] or exit_metadata['ExitCode'] > 0:
            raise RunnerException(command=command, stdout_log=stdout_log, stderr_log=stderr_log)
        return stdout_log, stderr_log

    def _get_volumes_and_docker_path(self) -> tuple[dict, str]:
        app = ConanApp(self.conan_api)
        remotes = self.conan_api.remotes.list(self.args.remote) if not self.args.no_remote else []
        conanfile = app.loader.load_consumer(self.abs_host_path / "conanfile.py", remotes=remotes)
        abs_docker_base_path = Path('/') / self.docker_user_name / 'conanrunner'
        # Check if recipe has defined a root folder
        # In this case, mount the root folder as the base path and update the abs_docker_path to the
        # new relative path
        if hasattr(conanfile, "layout"):
            try:
                conanfile.layout()
                if conanfile.folders.root:
                    abs_path = self._get_abs_host_path(conanfile.folders.root)
                    if self.abs_host_path.is_relative_to(abs_path):
                        abs_docker_base_path /= abs_path.name
                        volumes = {abs_path: {'bind': abs_docker_base_path.as_posix(), 'mode': 'rw'}}
                        abs_docker_path = abs_docker_base_path / self.abs_host_path.relative_to(abs_path)
                        return volumes, abs_docker_path.as_posix()
            except:
                pass
        abs_docker_path = (abs_docker_base_path / self.abs_host_path.name).as_posix()
        volumes = {self.abs_host_path: {'bind': abs_docker_path, 'mode': 'rw'}}
        return volumes, abs_docker_path

    def _create_runner_environment(self) -> tuple[dict, dict]:
        # Runner configuration
        self.abs_runner_home_path = self.abs_host_path / '.conanrunner'
        self.docker_user_name = self.configfile.run.user or 'root'
        volumes, self.abs_docker_path = self._get_volumes_and_docker_path()
        shutil.rmtree(self.abs_runner_home_path, ignore_errors=True)
        environment = {'CONAN_RUNNER_ENVIRONMENT': '1'}

        # Update conan command and some paths to run inside the container
        self.raw_args[self.raw_args.index(self.args.path)] = self.abs_docker_path
        self.command = ' '.join([f'conan {self.command}'] + [f'"{raw_arg}"' if ' ' in raw_arg else raw_arg for raw_arg in self.raw_args] + ['-f json > create.json'])

        if self.cache == 'shared':
            volumes[self.conan_api.home_folder] = {'bind': f'/{self.docker_user_name}/.conan2', 'mode': 'rw'}

        if self.cache in ['clean', 'copy']:
            # Copy all conan profiles and config files to docker workspace
            for profile in set(self.args.profile_host + (self.args.profile_build or [])):
                profile_path = self.conan_api.profiles.get_path(profile)
                dest_filename = self.abs_runner_home_path / 'profiles' / Path(profile)
                if not dest_filename.exists():
                    dest_filename.parent.mkdir(parents=True, exist_ok=True)
                self.logger.verbose(f"Copying profile '{profile}': {profile_path} -> {dest_filename}")
                shutil.copy(profile_path, dest_filename)
            if not self.args.profile_build:
                dest_filename = self.abs_runner_home_path / 'profiles' / "default"
                default_build_profile = self.conan_api.profiles.get_default_build()
                self.logger.verbose(f"Copying default profile: {default_build_profile} -> {dest_filename}")
                shutil.copy(default_build_profile, dest_filename.as_posix())

            for file_name in ['global.conf', 'settings.yml', 'remotes.json']:
                src_file = Path(self.conan_api.home_folder) / file_name
                if src_file.exists():
                    self.logger.verbose(f"Copying {src_file} -> {self.abs_runner_home_path / file_name}")
                    shutil.copy(src_file, self.abs_runner_home_path / file_name)

            if self.cache == 'copy':
                tgz_path = self.abs_runner_home_path / 'local_cache_save.tgz'
                self.logger.status(f'Save host cache in: {tgz_path}')
                self.conan_api.cache.save(self.conan_api.list.select(ListPattern("*:*")), tgz_path)
        return volumes, environment

    def _init_container(self) -> None:
        min_conan_version = '2.1'
        stdout, _ = self._run_command('conan --version', verbose=True)
        docker_conan_version = str(stdout.split('Conan version ')[1].replace('\n', '').replace('\r', '')) # Remove all characters and color
        if Version(docker_conan_version) <= Version(min_conan_version):
            raise ConanException(f'conan version inside the container must be greater than {min_conan_version}')
        if self.cache != 'shared':
            self._run_command('mkdir -p ${HOME}/.conan2/profiles', verbose=False)
            self._run_command('cp -r "'+self.abs_docker_path+'/.conanrunner/profiles/." ${HOME}/.conan2/profiles/.', verbose=False)

            for file_name in ['global.conf', 'settings.yml', 'remotes.json']:
                if (self.abs_runner_home_path / file_name).exists():
                    self._run_command('cp "'+self.abs_docker_path+'/.conanrunner/'+file_name+'" ${HOME}/.conan2/'+file_name, verbose=False)
            if self.cache in ['copy']:
                self._run_command('conan cache restore "'+self.abs_docker_path+'/.conanrunner/local_cache_save.tgz"')

    def _update_local_cache(self) -> None:
        if self.cache != 'shared':
            self._run_command('conan list --graph=create.json --graph-binaries=build --format=json > pkglist.json', verbose=False)
            self._run_command('conan cache save --list=pkglist.json --file "'+self.abs_docker_path+'"/.conanrunner/docker_cache_save.tgz')
            tgz_path = self.abs_runner_home_path / 'docker_cache_save.tgz'
            self.logger.status(f'Restore host cache from: {tgz_path}')
            self.conan_api.cache.restore(tgz_path)
