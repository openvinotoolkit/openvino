import itertools
import glob
import re

from conan.internal import check_duplicated_generator
from conan.internal.util.files import save
from conan.tools.premake.constants import CONAN_TO_PREMAKE_ARCH

# Filename format strings
PREMAKE_VAR_FILE = "conan_{pkgname}_vars_{config}.premake5.lua"
PREMAKE_PKG_FILE = "conan_{pkgname}.premake5.lua"
PREMAKE_ROOT_FILE = "conandeps.premake5.lua"

PREMAKE_CONFIG_FILE = "conanconfig_{config}.premake5.lua"
PREMAKE_CONFIG_ROOT_FILE = "conanconfig.premake5.lua"

# File template format strings
PREMAKE_TEMPLATE_CONFIG = """
include "conanutils.premake5.lua"

t_conan_deps_order = {{}}
t_conan_deps_order["{config}"] = {{{order}}}

if conan_deps_order == nil then conan_deps_order = {{}} end
conan_premake_tmerge(conan_deps_order, t_conan_deps_order)
"""
PREMAKE_TEMPLATE_UTILS = """
function conan_premake_tmerge(dst, src)
    for k, v in pairs(src) do
        if type(v) == "table" then
            if type(dst[k] or 0) == "table" then
                conan_premake_tmerge(dst[k] or {}, src[k] or {})
            else
                dst[k] = v
            end
        else
            dst[k] = v
        end
    end
    return dst
end
"""
PREMAKE_TEMPLATE_VAR = """
include "conanutils.premake5.lua"

t_conandeps = {{}}
t_conandeps["{config}"] = {{}}
t_conandeps["{config}"]["{pkgname}"] = {{}}
t_conandeps["{config}"]["{pkgname}"]["includedirs"] = {{{deps.includedirs}}}
t_conandeps["{config}"]["{pkgname}"]["libdirs"] = {{{deps.libdirs}}}
t_conandeps["{config}"]["{pkgname}"]["bindirs"] = {{{deps.bindirs}}}
t_conandeps["{config}"]["{pkgname}"]["libs"] = {{{deps.libs}}}
t_conandeps["{config}"]["{pkgname}"]["system_libs"] = {{{deps.system_libs}}}
t_conandeps["{config}"]["{pkgname}"]["defines"] = {{{deps.defines}}}
t_conandeps["{config}"]["{pkgname}"]["cxxflags"] = {{{deps.cxxflags}}}
t_conandeps["{config}"]["{pkgname}"]["cflags"] = {{{deps.cflags}}}
t_conandeps["{config}"]["{pkgname}"]["sharedlinkflags"] = {{{deps.sharedlinkflags}}}
t_conandeps["{config}"]["{pkgname}"]["exelinkflags"] = {{{deps.exelinkflags}}}
t_conandeps["{config}"]["{pkgname}"]["frameworks"] = {{{deps.frameworks}}}

if conandeps == nil then conandeps = {{}} end
conan_premake_tmerge(conandeps, t_conandeps)
"""
PREMAKE_TEMPLATE_ROOT_BUILD = """
        includedirs(conandeps[conf][pkg]["includedirs"])
        bindirs(conandeps[conf][pkg]["bindirs"])
        defines(conandeps[conf][pkg]["defines"])
"""
PREMAKE_TEMPLATE_ROOT_LINK = """
        libdirs(conandeps[conf][pkg]["libdirs"])
        links(conandeps[conf][pkg]["libs"])
        links(conandeps[conf][pkg]["system_libs"])
        links(conandeps[conf][pkg]["frameworks"])
"""
PREMAKE_TEMPLATE_ROOT_FUNCTION = """
function {function_name}(conf, pkg)
    if conf == nil then
{filter_call}
    elseif pkg == nil then
        local order = conan_deps_order[conf]
        for index, lib in ipairs(order) do
            {function_name}(conf, lib)
        end
    else
{lua_content}
    end
end
"""
PREMAKE_TEMPLATE_ROOT_GLOBAL = """
function conan_setup(conf, pkg)
    conan_setup_build(conf, pkg)
    conan_setup_link(conf, pkg)
end
"""


# Helper class that expands cpp_info meta information in lua readable string sequences
class _PremakeTemplate(object):
    def __init__(self, req, dep_cpp_info):
        def _format_paths(paths):
            if not paths:
                return ""
            return ",\n".join(f'"{p}"'.replace("\\", "/") for p in paths)

        def _format_flags(flags):
            if not flags:
                return ""
            return ", ".join('"%s"' % p.replace('"', '\\"') for p in flags)

        # Headers dependant
        with_headers = req and req.headers
        self.includedirs = _format_paths(dep_cpp_info.includedirs if with_headers else [])
        self.defines = _format_flags(dep_cpp_info.defines if with_headers else [])
        self.cxxflags = _format_flags(dep_cpp_info.cxxflags if with_headers else [])
        self.cflags = _format_flags(dep_cpp_info.cflags if with_headers else [])
        self.sharedlinkflags = _format_flags(dep_cpp_info.sharedlinkflags if with_headers else [])
        self.exelinkflags = _format_flags(dep_cpp_info.exelinkflags if with_headers else [])

        # Libs dependant
        with_libs = req and req.libs
        self.libs = _format_flags(dep_cpp_info.libs if with_libs else [])
        self.libdirs = _format_paths(dep_cpp_info.libdirs if with_libs else [])

        # Run dependant
        with_run = req and req.run
        self.bindirs = _format_paths(dep_cpp_info.bindirs if with_run else [])

        self.system_libs = _format_flags(dep_cpp_info.system_libs)
        self.frameworks = ", ".join('"%s.framework"' % p.replace('"', '\\"') for p in
                                    dep_cpp_info.frameworks) if dep_cpp_info.frameworks else ""
        self.sysroot = f"{dep_cpp_info.sysroot}".replace("\\", "/") \
            if dep_cpp_info.sysroot else ""


class PremakeDeps:
    """
    PremakeDeps class generator
    conandeps.premake5.lua: unconditional import of all *direct* dependencies only
    """

    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """

        self._conanfile = conanfile

        # Tab configuration
        self.tab = "    "

        # Return value buffer
        self.output_files = {}
        # Extract configuration and architecture form conanfile
        self.configuration = conanfile.settings.build_type
        self.architecture = conanfile.settings.arch

    def generate(self):
        """
        Generates ``conan_<pkg>_vars_<config>.premake5.lua``, ``conan_<pkg>_<config>.premake5.lua``,
        and ``conan_<pkg>.premake5.lua`` files into the ``conanfile.generators_folder``.
        """

        check_duplicated_generator(self, self._conanfile)
        # Current directory is the generators_folder
        generator_files = self.content
        for generator_file, content in generator_files.items():
            save(generator_file, content)

    def _config_suffix(self):
        return f"{self.configuration}_{CONAN_TO_PREMAKE_ARCH[str(self.architecture)]}".lower()

    def _output_lua_file(self, filename, content):
        self.output_files[filename] = "\n".join(["#!lua", *content])

    def _indent_string(self, string, indent=1):
        return "\n".join([
            f"{self.tab * indent}{line}" for line in list(filter(None, string.splitlines()))
        ])

    def _premake_filtered(self, content, configuration, architecture, indent=0):
        """
        - Surrounds the lua line(s) contained within ``content`` with a premake "filter" and returns the result.
        - A "filter" will affect all premake function calls after it's set. It's used to limit following project
          setup function call(s) to a certain scope. Here it is used to limit the calls in content to only apply
          if the premake ``configuration`` and ``architecture`` matches the parameters in this function call.
        """
        lines = list(itertools.chain.from_iterable([cnt.splitlines() for cnt in content]))
        return [
            # Set new filter
            f'{self.tab * indent}filter {{ "configurations:{configuration}", "architecture:{architecture}" }}',
            # Emit content
            *[f"{self.tab * indent}{self.tab}{line.strip()}" for line in list(filter(None, lines))],
            # Clear active filter
            f"{self.tab * indent}filter {{}}",
        ]

    @property
    def content(self):
        check_duplicated_generator(self, self._conanfile)

        self.output_files = {}
        conf_name = self._config_suffix()

        # Global utility file
        self._output_lua_file("conanutils.premake5.lua", [PREMAKE_TEMPLATE_UTILS])

        # Extract all dependencies in topological order: some linkers like ld or gold prunes the
        # functions which are not being used in the lookup table. If the less dependant libraries are
        # passed first, the linker will not be able to resolve the symbols in the dependent libraries
        # as they will have been removed
        host_req = self._conanfile.dependencies.host.topological_sort
        test_req = self._conanfile.dependencies.test.topological_sort
        build_req = self._conanfile.dependencies.direct_build.topological_sort

        # Merge into one list
        full_req = list(host_req.items()) + list(test_req.items()) + list(build_req.items())

        # Process dependencies and accumulate globally required data
        pkg_files = []
        dep_names = []
        config_sets = []
        for require, dep in full_req:
            dep_name = require.ref.name
            dep_names.append(dep_name)

            # Convert and aggregate dependency's
            dep_aggregate = dep.cpp_info.aggregated_components()

            # Generate config dependent package variable and setup premake file
            var_filename = PREMAKE_VAR_FILE.format(pkgname=dep_name, config=conf_name)
            self._output_lua_file(var_filename, [
                PREMAKE_TEMPLATE_VAR.format(pkgname=dep_name,
                    config=conf_name, deps=_PremakeTemplate(require, dep_aggregate))
            ])

            # Create list of all available profiles by searching on disk
            file_pattern = PREMAKE_VAR_FILE.format(pkgname=dep_name, config="*")
            file_regex = PREMAKE_VAR_FILE.format(pkgname=re.escape(dep_name), config="(([^_]*)_(.*))")
            available_config_files = glob.glob(file_pattern)
            # Add filename of current generations var file if not already present
            if var_filename not in available_config_files:
                available_config_files.append(var_filename)
            profiles = [
                (regex_res[0], regex_res.group(1), regex_res.group(2), regex_res.group(3)) for regex_res in [
                    re.search(file_regex, file_name) for file_name in available_config_files
                ]
            ]
            config_sets = [profile[1] for profile in profiles]

            # Emit package premake file
            pkg_filename = PREMAKE_PKG_FILE.format(pkgname=dep_name)
            pkg_files.append(pkg_filename)
            self._output_lua_file(pkg_filename, [
                # Includes
                *['include "{}"'.format(profile[0]) for profile in profiles],
            ])

        # Output global premake file
        self._output_lua_file(PREMAKE_ROOT_FILE, [
            # Includes
            *[f'include "{pkg_file}"' for pkg_file in pkg_files],
            # Global order for each configuration
            'include "conanconfig.premake5.lua"',
            # Functions
            PREMAKE_TEMPLATE_ROOT_FUNCTION.format(
                function_name="conan_setup_build",
                lua_content=PREMAKE_TEMPLATE_ROOT_BUILD,
                filter_call="\n".join(
                    ["\n".join(self._premake_filtered(
                        [f'conan_setup_build("{config}")'], config.split("_", 1)[0], config.split("_", 1)[1], 2)
                    ) for config in config_sets]
                )
            ),
            PREMAKE_TEMPLATE_ROOT_FUNCTION.format(
                function_name="conan_setup_link",
                lua_content=PREMAKE_TEMPLATE_ROOT_LINK,
                filter_call="\n".join(
                    ["\n".join(self._premake_filtered(
                        [f'conan_setup_link("{config}")'], config.split("_", 1)[0], config.split("_", 1)[1], 2)
                    ) for config in config_sets]
                )
            ),
            PREMAKE_TEMPLATE_ROOT_GLOBAL
        ])

        # Output configuration file for the current build configuration
        self._output_lua_file(PREMAKE_CONFIG_FILE.format(config=conf_name), [
            PREMAKE_TEMPLATE_CONFIG.format(
                config=conf_name, order=", ".join(f'"{name}"' for name in reversed(dep_names))
            )
        ])

        # Output root configuration file
        available_config_files = glob.glob(PREMAKE_CONFIG_FILE.format(config="*"))
        available_configs = [file_name.split("_", 1)[1].split(".")[0] for file_name in available_config_files]
        available_configs.append(conf_name)
        self._output_lua_file(PREMAKE_CONFIG_ROOT_FILE, [
            *['include "{}"'.format(PREMAKE_CONFIG_FILE.format(config=config)) for config in available_configs],
        ])

        return self.output_files
