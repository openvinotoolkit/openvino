
from conan.internal import check_duplicated_generator
from conan.tools.build.flags import build_type_flags, cppstd_flag, build_type_link_flags
from conan.tools.env import Environment
from conan.tools.microsoft.visual import msvc_runtime_flag, VCVars


class NMakeToolchain:
    """
    https://learn.microsoft.com/en-us/cpp/build/reference/running-nmake?view=msvc-170#toolsini-and-nmake
    We have also explored the usage of Tools.ini:
    https://learn.microsoft.com/en-us/cpp/build/reference/running-nmake?view=msvc-170
    but not possible, because it cannot include other files, it will also potentially collide with
    a user Tool.ini, without easy resolution. At least the environment is additive.
    """
    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile

        # Flags
        self.extra_cflags = []
        self.extra_cxxflags = []
        self.extra_ldflags = []
        self.extra_defines = []

    @staticmethod
    def _format_options(options):
        return [f"{opt[0].replace('-', '/')}{opt[1:]}" for opt in options if len(opt) > 1]

    @staticmethod
    def _format_defines(defines):
        formated_defines = []
        for define in defines:
            if "=" in define:
                # CL env-var can't accept '=' sign in /D option, it can be replaced by '#' sign:
                # https://learn.microsoft.com/en-us/cpp/build/reference/cl-environment-variables
                macro, value = define.split("=", 1)
                if value and not value.isnumeric():
                    value = f'\\"{value}\\"'
                define = f"{macro}#{value}"
            formated_defines.append(f"/D\"{define}\"")
        return formated_defines

    @property
    def _cl(self):
        bt_flags = build_type_flags(self._conanfile)
        bt_flags = bt_flags if bt_flags else []

        rt_flags = msvc_runtime_flag(self._conanfile)
        rt_flags = [f"/{rt_flags}"] if rt_flags else []

        cflags = []
        cflags.extend(self._conanfile.conf.get("tools.build:cflags", default=[], check_type=list))
        cflags.extend(self.extra_cflags)

        cxxflags = []
        cppstd = cppstd_flag(self._conanfile)
        if cppstd:
            cxxflags.append(cppstd)
        cxxflags.extend(self._conanfile.conf.get("tools.build:cxxflags", default=[],
                                                 check_type=list))
        cxxflags.extend(self.extra_cxxflags)

        defines = []
        build_type = self._conanfile.settings.get_safe("build_type")
        if build_type in ["Release", "RelWithDebInfo", "MinSizeRel"]:
            defines.append("NDEBUG")
        defines.extend(self._conanfile.conf.get("tools.build:defines", default=[], check_type=list))
        defines.extend(self.extra_defines)

        return (["/nologo"] + self._format_options(bt_flags + rt_flags + cflags + cxxflags) +
                self._format_defines(defines))

    @property
    def _link(self):
        bt_ldflags = build_type_link_flags(self._conanfile.settings)
        bt_ldflags = bt_ldflags if bt_ldflags else []

        ldflags = []
        ldflags.extend(bt_ldflags)
        ldflags.extend(self._conanfile.conf.get("tools.build:sharedlinkflags", default=[],
                                                check_type=list))
        ldflags.extend(self._conanfile.conf.get("tools.build:exelinkflags", default=[],
                                                check_type=list))
        ldflags.extend(self.extra_ldflags)

        return ["/nologo"] + self._format_options(ldflags)

    def environment(self):
        env = Environment()
        # Injection of compile flags in CL env-var:
        # https://learn.microsoft.com/en-us/cpp/build/reference/cl-environment-variables
        env.append("CL", self._cl)
        # Injection of link flags in _LINK_ env-var:
        # https://learn.microsoft.com/en-us/cpp/build/reference/linking
        env.append("_LINK_", self._link)
        # Also define some special env-vars which can override special NMake macros:
        # https://learn.microsoft.com/en-us/cpp/build/reference/special-nmake-macros
        conf_compilers = self._conanfile.conf.get("tools.build:compiler_executables", default={},
                                                  check_type=dict)
        if conf_compilers:
            compilers_mapping = {
                "AS": "asm",
                "CC": "c",
                "CPP": "cpp",
                "CXX": "cpp",
                "RC": "rc",
            }
            for env_var, comp in compilers_mapping.items():
                if comp in conf_compilers:
                    env.define(env_var, conf_compilers[comp])
        return env

    def vars(self):
        return self.environment().vars(self._conanfile, scope="build")

    def generate(self, env=None, scope="build"):
        check_duplicated_generator(self, self._conanfile)
        env = env or self.environment()
        env.vars(self._conanfile, scope=scope).save_script("conannmaketoolchain")
        VCVars(self._conanfile).generate(scope=scope)
