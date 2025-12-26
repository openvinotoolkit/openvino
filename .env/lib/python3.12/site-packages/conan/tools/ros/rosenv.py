import os
from conan.api.output import Color
from conan.tools.env import Environment
from conan.tools.env.environment import create_env_script


class ROSEnv:
    """
    Generator to serve as integration for Robot Operating System 2 development workspaces.

    IMPORTANT: This generator should be used together with CMakeDeps and CMakeToolchain generators.
    """

    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile
        self.variables = {}
        self._build_script_sh_file = "conanrosenv-build.sh"
        self._build_script_bat_file = "conanrosenv-build.bat"
        self._wrapper_script_sh_file = "conanrosenv.sh"
        self._wrapper_script_bat_file = "conanrosenv.bat"

    def generate(self):
        """
        Creates a ``conanrosenv.sh`` with the environment variables that are needed to build and
        execute ROS packages with Conan dependencies.
        """
        cmake_toolchain_path = os.path.join(self._conanfile.generators_folder,
                                            "conan_toolchain.cmake")
        self.variables["CMAKE_TOOLCHAIN_FILE"] = cmake_toolchain_path
        build_type = self._conanfile.settings.get_safe("build_type")
        if build_type:
            self.variables["CMAKE_BUILD_TYPE"] = build_type

        # Add ROS required variables to VirtualBuildEnv
        rosbuildenv = Environment()
        for k, v in self.variables.items():
            rosbuildenv.define(k, v)
        for build_script_file in [self._build_script_sh_file, self._build_script_bat_file]:
            rosbuildenv.vars(self._conanfile, "build").save_script(build_script_file)

        #TODO: Add powrshell support generating .ps1 files
        self._generate_sh_files()
        self._generate_bat_files()

    def _generate_sh_files(self):
        # Generate conanrosenv.sh script wrapper that calls conanbuild.sh and conanrun.sh
        conanbuild_sh_path = os.path.join(self._conanfile.generators_folder, "conanbuild.sh")
        conanrun_sh_path = os.path.join(self._conanfile.generators_folder, "conanrun.sh")
        rosenv_sh_wrapper_content = [f". \"{conanbuild_sh_path}\"", f". \"{conanrun_sh_path}\""]
        create_env_script(self._conanfile, "\n".join(rosenv_sh_wrapper_content),
                          self._wrapper_script_sh_file, None)

        conanrosenv_path = os.path.join(self._conanfile.generators_folder,
                                        self._wrapper_script_sh_file)
        msg = f"Generated ROSEnv Conan file: {self._wrapper_script_sh_file}\n" + \
              f"Use 'source {conanrosenv_path}' to set the ROSEnv Conan before 'colcon build'"
        self._conanfile.output.info(msg, fg=Color.CYAN)

    def _generate_bat_files(self):
        # Generate conanrosenv.bat script wrapper that calls conanbuild.bat and conanrun.bat
        conanbuild_bat_path = os.path.join(self._conanfile.generators_folder, "conanbuild.bat")
        conanrun_bat_path = os.path.join(self._conanfile.generators_folder, "conanrun.bat")
        rosenv_bat_wrapper_content = [
            "@echo off",
            f"CALL \"{conanbuild_bat_path}\"",
            f"CALL \"{conanrun_bat_path}\""
            ]
        create_env_script(self._conanfile, "\n".join(rosenv_bat_wrapper_content),
                          self._wrapper_script_bat_file, None)

        conanrosenv_path = os.path.join(self._conanfile.generators_folder,
                                        self._wrapper_script_bat_file)
        msg = f"Generated ROSEnv Conan file: {self._wrapper_script_bat_file}\n" + \
              f"Use 'call {conanrosenv_path}' to set the ROSEnv Conan before 'colcon build'"
        self._conanfile.output.info(msg, fg=Color.CYAN)
