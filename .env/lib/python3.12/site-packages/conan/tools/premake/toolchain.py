from conan.tools.build.flags import architecture_flag, architecture_link_flag, libcxx_flags, threads_flags
import os
import textwrap
from pathlib import Path

from conan.tools.env.virtualbuildenv import VirtualBuildEnv
from jinja2 import Template

from conan.tools.build.cross_building import cross_building
from conan.tools.files import save
from conan.tools.microsoft.visual import VCVars
from conan.tools.premake.premakedeps import PREMAKE_ROOT_FILE


def _generate_flags(self, conanfile):
    template = textwrap.dedent(
        """\
        {% if extra_cflags %}
        -- C flags retrieved from CFLAGS environment, conan.conf(tools.build:cflags), extra_cflags and compiler settings
        filter { files { "**.c" } }
            buildoptions { {{ extra_cflags }} }
        filter {}
        {% endif %}
        {% if extra_cxxflags %}
        -- CXX flags retrieved from CXXFLAGS environment, conan.conf(tools.build:cxxflags), extra_cxxflags and compiler settings
        filter { files { "**.cpp", "**.cxx", "**.cc" } }
            buildoptions { {{ extra_cxxflags }} }
        filter {}
        {% endif %}
        {% if extra_ldflags %}
        -- Link flags retrieved from LDFLAGS environment, conan.conf(tools.build:sharedlinkflags), conan.conf(tools.build:exelinkflags), extra_cxxflags and compiler settings
        linkoptions { {{ extra_ldflags }} }
        {% endif %}
        {% if extra_defines %}
        -- Defines retrieved from DEFINES environment, conan.conf(tools.build:defines) and extra_defines
        defines { {{ extra_defines }} }
        {% endif %}
    """
    )

    def format_list(items):
        return ", ".join(f'"{item}"' for item in items) if items else None

    def to_list(value):
        return value if isinstance(value, list) else [value] if value else []

    arch_flags = to_list(architecture_flag(self._conanfile))
    cxx_flags, libcxx_compile_definitions = libcxx_flags(self._conanfile)
    arch_link_flags = to_list(architecture_link_flag(self._conanfile))
    thread_flags_list = threads_flags(self._conanfile)

    extra_defines = format_list(
        conanfile.conf.get("tools.build:defines", default=[], check_type=list)
        + self.extra_defines
        + to_list(libcxx_compile_definitions)
    )
    extra_c_flags = format_list(
        conanfile.conf.get("tools.build:cflags", default=[], check_type=list)
        + self.extra_cflags
        + arch_flags
        + thread_flags_list
    )
    extra_cxx_flags = format_list(
        conanfile.conf.get("tools.build:cxxflags", default=[], check_type=list)
        + to_list(cxx_flags)
        + self.extra_cxxflags
        + arch_flags
        + thread_flags_list
    )
    extra_ld_flags = format_list(
        conanfile.conf.get("tools.build:sharedlinkflags", default=[], check_type=list)
        + conanfile.conf.get("tools.build:exelinkflags", default=[], check_type=list)
        + self.extra_ldflags
        + arch_flags
        + arch_link_flags
        + thread_flags_list
    )

    return (
        Template(template, trim_blocks=True, lstrip_blocks=True)
        .render(
            extra_defines=extra_defines,
            extra_cflags=extra_c_flags,
            extra_cxxflags=extra_cxx_flags,
            extra_ldflags=extra_ld_flags,
        )
        .strip()
    )


class _PremakeProject:
    _premake_project_template = textwrap.dedent(
        """\
    project "{{ name }}"
        {% if kind %}
        kind "{{ kind }}"
        {% endif %}
        {% if flags %}
    {{ flags | indent(indent_level, first=True) }}
        {% endif %}
    """
    )

    def __init__(self, name, conanfile) -> None:
        self.name = name
        self.kind = None
        self.extra_cxxflags = []
        self.extra_cflags = []
        self.extra_ldflags = []
        self.extra_defines = []
        self.disable = False
        self._conanfile = conanfile

    def _generate(self):
        """Generates project block"""
        flags_content = _generate_flags(self, self._conanfile)  # Generate flags specific to this project
        return Template(self._premake_project_template, trim_blocks=True, lstrip_blocks=True).render(
            name=self.name,
            kind="None" if self.disable else self.kind,
            flags=flags_content,
            indent_level=4,
        )


class PremakeToolchain:
    """
    PremakeToolchain generator
    """

    filename = "conantoolchain.premake5.lua"
    # Keep template indented correctly for Lua output
    _premake_file_template = textwrap.dedent(
        """\
    #!lua
    -- Conan auto-generated toolchain file
    {% if has_conan_deps %}
    -- Include conandeps.premake5.lua with Conan dependency setup
    include("conandeps.premake5.lua")
    {% endif %}

    -- Base build directory
    local locationDir = path.normalize("{{ build_folder }}")

    -- Generate workspace configurations
    for wks in premake.global.eachWorkspace() do
        workspace(wks.name)
            -- Set base location for all workspaces
            location(locationDir)
            targetdir(path.join(locationDir, "bin"))
            objdir(path.join(locationDir, "obj"))

            {% if cppstd %}
            cppdialect "{{ cppstd }}"
            {% endif %}
            {% if cstd %}
            cdialect "{{ cstd }}"
            {% endif %}
            {% if shared != None %}
            -- IMPORTANT: this global setting will only apply `project`s which do not have `kind` set.
            -- IMPORTANT: This will not override existing `kind` set in `project` block.
            -- To let conan take control over `kind` of the libraries, DO NOT SET `kind` (StaticLib or
            -- SharedLib) in `project` block.
            kind "{{ "SharedLib" if shared else "StaticLib" }}"
            {% endif %}
            {% if fpic != None %}
            -- Enable position independent code
            pic "{{ "On" if fpic else "Off" }}"
            {% endif %}
            filter { "architecture: not wasm64" }
                -- TODO: There is an issue with premake and "wasm64" when system is declared "emscripten"
                system "{{ target_build_os }}"
            filter {}
            {% if macho_to_amd64 %}
            -- TODO: this should be fixed by premake: https://github.com/premake/premake-core/issues/2136
            buildoptions "-arch x86_64"
            linkoptions "-arch x86_64"
            {% endif %}
            {% if target_build_os == "emscripten" %}
            filter { "system:emscripten", "kind:ConsoleApp or WindowedApp" }
                -- Replace built in .wasm extension to .js to generate also a JavaScript files
                targetextension ".js"
            filter {}
            {% endif %}
            {% if flags %}
    {{ flags | indent(indent_level, first=True) }}
            {% endif %}

            filter { "system:macosx" }
                -- SHARED LIBS
                -- In the future we could add an opt in configuration to run
                -- fix_apple_shared_install_name on executables to have a similar behavior as CMake
                -- generator. Premake does not allow adding absolute RCPATHS
                -- Due to this limitation, if a consumer depends on a premake shared recipe, it will
                -- require to run conanrun script to setup proper DYLD_LIBRARY_PATH
                -- Reference: https://github.com/premake/premake-core/issues/2262#issuecomment-2378250385
                linkoptions { "-Wl,-rpath,@loader_path" }
            filter {}

            conan_setup()
    end

        {% for project in projects.values() %}

    {{ project._generate() }}
        {% endfor %}
    """
    )

    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile
        self._projects = {}
        # Extra flags
        #: List of extra ``CXX`` flags. Added to ``buildoptions``.
        self.extra_cxxflags = []
        #: List of extra ``C`` flags. Added to ``buildoptions``.
        self.extra_cflags = []
        #: List of extra linker flags. Added to ``linkoptions``.
        self.extra_ldflags = []
        #: List of extra preprocessor definitions. Added to ``defines``.
        self.extra_defines = []

    def project(self, project_name):
        """
        The returned object will also have the same properties as the workspace but will only affect
        the project with the name.
        :param project_name: The name of the project inside the workspace to be updated.
        :return: ``<PremakeProject>`` object which allow to set project specific flags.
        """
        if project_name not in self._projects:
            self._projects[project_name] = _PremakeProject(project_name, self._conanfile)
        return self._projects[project_name]

    def generate(self):
        """
        Creates a ``conantoolchain.premake5.lua`` file which will properly configure build paths,
        binary paths, configuration settings and compiler/linker flags based on toolchain
        configuration.
        """
        premake_conan_deps = Path(self._conanfile.generators_folder) / PREMAKE_ROOT_FILE
        cppstd = self._conanfile.settings.get_safe("compiler.cppstd")
        if cppstd:
            # See premake possible cppstd values: https://premake.github.io/docs/cppdialect/
            if cppstd.startswith("gnu"):
                cppstd = f"gnu++{cppstd[3:]}"
            elif cppstd[0].isnumeric():
                cppstd = f"c++{cppstd}"

        compilers_build_mapping = self._conanfile.conf.get(
            "tools.build:compiler_executables", default={}, check_type=dict
        )
        if compilers_build_mapping:
            build_env = VirtualBuildEnv(self._conanfile, auto_generate=False)
            env = build_env.environment()
            if "c" in compilers_build_mapping:
                env.define("CC", compilers_build_mapping["c"])
            if "cpp" in compilers_build_mapping:
                env.define("CXX", compilers_build_mapping["cpp"])
            build_env.generate()

        macho_to_amd64 = (
            self._conanfile.settings.arch
            if cross_building(self._conanfile) and self._conanfile.settings.os == "Macos"
            else None
        )

        content = Template(self._premake_file_template, trim_blocks=True, lstrip_blocks=True).render(
            # Pass posix path for better cross-platform compatibility in Lua
            build_folder=Path(self._conanfile.build_folder).as_posix(),
            has_conan_deps=premake_conan_deps.exists(),
            cppstd=cppstd,
            cstd=self._conanfile.settings.get_safe("compiler.cstd"),
            shared=self._conanfile.options.get_safe("shared"),
            fpic=self._conanfile.options.get_safe("fPIC"),
            target_build_os=self._target_build_os(),
            macho_to_amd64=macho_to_amd64,
            projects=self._projects,
            flags=_generate_flags(self, self._conanfile),
            indent_level=8,
        )
        save(
            self,
            os.path.join(self._conanfile.generators_folder, self.filename),
            content,
        )
        # Generate VCVars if using MSVC
        if "msvc" in self._conanfile.settings.compiler:
            VCVars(self._conanfile).generate()

    def _target_build_os(self):
        conan_os = str(self._conanfile.settings.os)
        if conan_os == "Macos":
            return "macosx"
        return conan_os.lower()
