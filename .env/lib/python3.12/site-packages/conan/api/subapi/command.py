import os
import shlex

from conan.api.output import ConanOutput
from conan.errors import ConanException


class CommandAPI:

    def __init__(self, conan_api):
        self._conan_api = conan_api
        self.cli = None

    def run(self, cmd):
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        if isinstance(cmd, list):
            current_cmd = cmd[0]
            args = cmd[1:]
        else:
            raise ConanException("Input of conan_api.command.run() should be a list or a string")
        commands = getattr(self.cli, "_commands")  # to no make it public to users of Cli class
        try:
            command = commands[current_cmd]
        except KeyError:
            raise ConanException(f"Command {current_cmd} does not exist")
        # Conan has some global state in the ConanOutput class that
        # get redefined when running a command and leak to the calling scope
        # if running from a custom command.
        # Store the old one and restore it after the command execution as a workaround.
        _conan_output_level = ConanOutput._conan_output_level  # noqa
        _silent_warn_tags = ConanOutput._silent_warn_tags  # noqa
        _warnings_as_errors = ConanOutput._warnings_as_errors  # noqa

        try:
            result = command.run_cli(self._conan_api, args)
        finally:
            ConanOutput._conan_output_level = _conan_output_level
            ConanOutput._silent_warn_tags = _silent_warn_tags
            ConanOutput._warnings_as_errors = _warnings_as_errors
        return result

    @staticmethod
    def get_runner(profile_host):
        if profile_host.runner and not os.environ.get("CONAN_RUNNER_ENVIRONMENT"):
            from conan.internal.runner.docker import DockerRunner
            from conan.internal.runner.ssh import SSHRunner
            from conan.internal.runner.wsl import WSLRunner
            try:
                runner_type = profile_host.runner['type'].lower()
            except KeyError:
                raise ConanException(f"Invalid runner configuration. 'type' must be defined")
            runner_instances_map = {
                'docker': DockerRunner,
                # 'ssh': SSHRunner,
                # 'wsl': WSLRunner,
            }
            try:
                runner_instance = runner_instances_map[runner_type]
            except KeyError:
                raise ConanException(f"Invalid runner type '{runner_type}'. "
                                     f"Allowed values: {', '.join(runner_instances_map.keys())}")
            return runner_instance
