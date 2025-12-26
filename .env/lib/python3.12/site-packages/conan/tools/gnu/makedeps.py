"""Makefile generator for Conan dependencies

This generator creates a Makefile (conandeps.mk) with variables for each dependency and consider their components.
To simplify its usage, it also creates global variables with aggregated values from all dependencies.
This generator does not work like a toolchain, it does not include settings.

For better customization, it allows appending prefixes as flags variables:

- CONAN_LIB_FLAG: Add a prefix to all libs variables, e.g. -l
- CONAN_DEFINE_FLAG: Add a prefix to all defines variables, e.g. -D
- CONAN_SYSTEM_LIB_FLAG: Add a prefix to all system_libs variables, e.g. -l
- CONAN_INCLUDE_DIR_FLAG: Add a prefix to all include dirs variables, e.g. -I
- CONAN_LIB_DIR_FLAG: Add a prefix to all lib dirs variables, e.g. -L
- CONAN_BIN_DIR_FLAG: Add a prefix to all bin dirs variables, e.g. -L


The conandeps.mk file layout is as follows:

- CONAN_DEPS: list all transitive and direct dependencies names without version (e.g. zlib)
- Iterate over each dependency and its components:
    - Prints name, version, root folder, regular folders, libs and flags
    - Components are rootified to avoid repeating same prefix twice for the root folder
    - Components libs, folder and flags are solved as variables to avoid repeating same name twice
- Aggregated global variables for simplification, sum all dependencies to common variables (e.g. CONAN_INCLUDE_DIRS)

"""

import os
import re
import textwrap

from jinja2 import Template, StrictUndefined
from typing import Optional

from conan.api.output import ConanOutput
from conan.internal import check_duplicated_generator
from conan.tools.files import save


CONAN_MAKEFILE_FILENAME = "conandeps.mk"


def _get_formatted_dirs(folders: list, prefix_path_: str, name: str) -> list:
    """ Format the directories to be used in the makefile, adding the prefix path if needed
    :param folders: list of directories
    :param prefix_path_: prefix path
    :param name: component name
    :return: list of formatted directories
    """
    ret = []
    for directory in folders:
        if directory.startswith("$(CONAN"):  # already a variable
            ret.append(directory)
            continue
        directory = os.path.normpath(directory).replace("\\", "/")
        prefix = ""
        if not os.path.isabs(directory):
            prefix = f"$(CONAN_ROOT_{name})/"
        elif directory.startswith(prefix_path_):
            prefix = f"$(CONAN_ROOT_{name})/"
            directory = os.path.relpath(directory, prefix_path_).replace("\\", "/")
        ret.append(f"{prefix}{directory}")
    return ret


def _makefy(name: str) -> str:
    """
    Convert a name to Make-variable-friendly syntax
    :param name: The name to be converted
    :return: Safe makefile variable, not including bad characters that are not parsed correctly
    """
    return re.sub(r'[^0-9A-Z_]', '_', name.upper())


def _makefy_properties(properties: Optional[dict]) -> dict:
    """
    Convert property dictionary keys to Make-variable-friendly syntax
    :param properties: The property dictionary to be converted (None is also accepted)
    :return: Modified property dictionary with keys not including bad characters that are not parsed correctly
    """
    return {_makefy(name): value for name, value in properties.items()} if properties else {}


def _check_property_value(name, value, output):
    if "\n" in value:
        output.warning(f"Skipping propery '{name}' because it contains newline")
        return False
    else:
        return True


def _filter_properties(properties: Optional[dict], output) -> dict:
    """
    Filter out properties whose values contain newlines, because they would break the generated makefile
    :param properties: A property dictionary (None is also accepted)
    :return: A property dictionary without the properties containing newlines
    """
    return {name: value for name, value in properties.items() if _check_property_value(name, value, output)} if properties else {}


def _conan_prefix_flag(variable: str) -> str:
    """
    Return a global flag to be used as prefix to any value in the makefile
    """
    return f"$(CONAN_{variable.upper()}_FLAG)" if variable else ""


def _common_cppinfo_variables() -> dict:
    """
    Regular cppinfo variables exported by any Conanfile and their Makefile prefixes
    """
    return {
        "objects": None,
        "libs": "lib",
        "defines": "define",
        "cflags": None,
        "cxxflags": None,
        "sharedlinkflags": None,
        "exelinkflags": None,
        "frameworks": None,
        "requires": None,
        "system_libs": "system_lib",
    }


def _common_cppinfo_dirs() -> dict:
    """
    Regular cppinfo folders exported by any Conanfile and their Makefile prefixes
    """
    return {
        "includedirs": "include_dir",
        "libdirs": "lib_dir",
        "bindirs": "bin_dir",
        "srcdirs": None,
        "builddirs": None,
        "resdirs": None,
        "frameworkdirs": None,
    }


def _jinja_format_list_values() -> str:
    """
    Template method to format a list of values in a Makefile,
    - Empty variables are not exposed in the Makefile
    - Single value variables are exposed in a single line
    - Multiple value variables are exposed in multiple lines with a tabulation
    e.g.
    define_variable_value("FOO", ["single_value"])
    output:
    FOO = single_value

    define_variable_value("BAR", ["value1", "value2"])
    output:
    BAR = \
        value1 \
        value2
    """
    return textwrap.dedent("""\
            {%- macro define_variable_value_safe(var, object, attribute) -%}
            {%- if attribute in object -%}
            {{ define_variable_value("{}".format(var), object[attribute]) }}
            {%- endif -%}
            {%- endmacro %}

            {%- macro define_multiple_variable_value(var, values) -%}
            {% for property_name, value in values.items() %}
            {{ var }}_{{ property_name }} = {{ value }}
            {% endfor %}
            {%- endmacro %}

            {%- macro define_variable_value(var, values) -%}
            {%- if values is not none -%}
            {%- if values|length > 0 -%}
            {{ var }} = {{ format_list_values(values) }}
            {%- endif -%}
            {%- endif -%}
            {%- endmacro %}

            {%- macro format_list_values(values) -%}
            {% if values|length == 1 %}
            {{ values[0] }}

            {% elif values|length > 1 %}
            \\
            {% for value in values[:-1] %}
            \t{{ value }} \\
            {% endfor %}
            \t{{ values|last }}

            {% endif %}
            {%- endmacro %}
            """)


class MakeInfo:
    """
    Store temporary information about each dependency
    """

    def __init__(self, name: str, dirs: list, flags: list):
        """
        :param name: Dependency or component raw name
        :param dirs: cpp_info folders supported by the dependency
        :param flags: cpp_info variables supported by the dependency
        """
        self._name = name
        self._dirs = dirs
        self._flags = flags

    @property
    def name(self) -> str:
        return self._name

    @property
    def dirs(self) -> list:
        """
        :return: List of cpp_info folders supported by the dependency without duplicates
        """
        return list(set(self._dirs))

    @property
    def flags(self) -> list:
        """
        :return: List of cpp_info variables supported by the dependency without duplicates
        """
        return list(set(self._flags))

    def dirs_append(self, directory: str):
        """
        Add a new cpp_info folder to the dependency
        """
        self._dirs.append(directory)

    def flags_append(self, flag: str):
        """
        Add a new cpp_info variable to the dependency
        """
        self._flags.append(flag)


class GlobalContentGenerator:
    """
    Generates the formatted content for global variables (e.g. CONAN_DEPS, CONAN_LIBS)
    """

    template = textwrap.dedent("""\

            # Aggregated global variables

            {{ define_variable_value("CONAN_INCLUDE_DIRS", deps_cpp_info_dirs.include_dirs) -}}
            {{- define_variable_value("CONAN_LIB_DIRS", deps_cpp_info_dirs.lib_dirs) -}}
            {{- define_variable_value("CONAN_BIN_DIRS", deps_cpp_info_dirs.bin_dirs) -}}
            {{- define_variable_value("CONAN_SRC_DIRS", deps_cpp_info_dirs.src_dirs) -}}
            {{- define_variable_value("CONAN_BUILD_DIRS", deps_cpp_info_dirs.build_dirs) -}}
            {{- define_variable_value("CONAN_RES_DIRS", deps_cpp_info_dirs.res_dirs) -}}
            {{- define_variable_value("CONAN_FRAMEWORK_DIRS", deps_cpp_info_dirs.framework_dirs) -}}
            {{- define_variable_value("CONAN_OBJECTS", deps_cpp_info_flags.objects) -}}
            {{- define_variable_value("CONAN_LIBS", deps_cpp_info_flags.libs) -}}
            {{- define_variable_value("CONAN_DEFINES", deps_cpp_info_flags.defines) -}}
            {{- define_variable_value("CONAN_CFLAGS", deps_cpp_info_flags.cflags) -}}
            {{- define_variable_value("CONAN_CXXFLAGS", deps_cpp_info_flags.cxxflags) -}}
            {{- define_variable_value("CONAN_SHAREDLINKFLAGS", deps_cpp_info_flags.sharedlinkflags) -}}
            {{- define_variable_value("CONAN_EXELINKFLAGS", deps_cpp_info_flags.exelinkflags) -}}
            {{- define_variable_value("CONAN_FRAMEWORKS", deps_cpp_info_flags.frameworks) -}}
            {{- define_variable_value("CONAN_REQUIRES", deps_cpp_info_flags.requires) -}}
            {{- define_variable_value("CONAN_SYSTEM_LIBS", deps_cpp_info_flags.system_libs) -}}
            """)

    template_deps = textwrap.dedent("""\
            {{ define_variable_value("CONAN_DEPS", deps) }}
            """)

    def content(self, deps_cpp_info_dirs: dict, deps_cpp_info_flags: dict) -> str:
        """
        Generate content for Cppinfo variables (e.g. CONAN_LIBS, CONAN_INCLUDE_DIRS)
        :param deps_cpp_info_dirs: Formatted dependencies folders
        :param deps_cpp_info_flags: Formatted dependencies variables
        """
        context = {"deps_cpp_info_dirs": deps_cpp_info_dirs,
                   "deps_cpp_info_flags": deps_cpp_info_flags}
        template = Template(_jinja_format_list_values() + self.template, trim_blocks=True,
                            lstrip_blocks=True, undefined=StrictUndefined)
        return template.render(context)

    def deps_content(self, dependencies_names: list) -> str:
        """
        Generate content for CONAN_DEPS (e.g. CONAN_DEPS = zlib, openssl)
        :param dependencies_names: Non-formatted dependencies names
        """
        context = {"deps": dependencies_names}
        template = Template(_jinja_format_list_values() + self.template_deps, trim_blocks=True,
                            lstrip_blocks=True, undefined=StrictUndefined)
        return template.render(context)


class GlobalGenerator:
    """
    Process all collected dependencies and parse to generate global content
    """

    def __init__(self, conanfile, make_infos):
        self._conanfile = conanfile
        self._make_infos = make_infos

    def _get_dependency_dirs(self) -> dict:
        """
        List regular directories from cpp_info and format them to be used in the makefile
        """
        dirs = {}
        for var in _common_cppinfo_dirs():
            key = var.replace("dirs", "_dirs")
            dirs[key] = [f"$(CONAN_{key.upper()}_{_makefy(makeinfo.name)})"
                         for makeinfo in self._make_infos if var in makeinfo.dirs]
        return dirs

    def _get_dependency_flags(self) -> dict:
        """
        List common variables from cpp_info and format them to be used in the makefile
        """
        flags = {}
        for var in _common_cppinfo_variables():
            key = var.replace("dirs", "_dirs")
            flags[key] = [f"$(CONAN_{key.upper()}_{_makefy(makeinfo.name)})"
                          for makeinfo in self._make_infos if var in makeinfo.flags]
        return flags

    def generate(self) -> str:
        """
        Process folder and variables for a dependency and generates its Makefile content
        """
        glob_content_gen = GlobalContentGenerator()
        dirs = self._get_dependency_dirs()
        flags = self._get_dependency_flags()
        return glob_content_gen.content(dirs, flags)

    def deps_generate(self) -> str:
        """
        Process dependencies names and generates its Makefile content.
        It should be added as first variable in the Makefile.
        """
        dependencies = [makeinfo.name for makeinfo in self._make_infos
                        if makeinfo.name != self._conanfile.name]
        glob_content_gen = GlobalContentGenerator()
        return glob_content_gen.deps_content(dependencies)


class DepComponentContentGenerator:
    """
    Generates Makefile content for each dependency component
    """

    template = textwrap.dedent("""\
        # {{ dep.ref.name }}::{{ comp_name }}

        {{  define_variable_value_safe("CONAN_INCLUDE_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'include_dirs') -}}
        {{- define_variable_value_safe("CONAN_LIB_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'lib_dirs') -}}
        {{- define_variable_value_safe("CONAN_BIN_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'bin_dirs') -}}
        {{- define_variable_value_safe("CONAN_SRC_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'src_dirs') -}}
        {{- define_variable_value_safe("CONAN_BUILD_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'build_dirs') -}}
        {{- define_variable_value_safe("CONAN_RES_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'res_dirs') -}}
        {{- define_variable_value_safe("CONAN_FRAMEWORK_DIRS_{}_{}".format(dep_name, name), cpp_info_dirs, 'framework_dirs') -}}
        {{- define_variable_value_safe("CONAN_OBJECTS_{}_{}".format(dep_name, name), cpp_info_flags, 'objects') -}}
        {{- define_variable_value_safe("CONAN_LIBS_{}_{}".format(dep_name, name), cpp_info_flags, 'libs') -}}
        {{- define_variable_value_safe("CONAN_DEFINES_{}_{}".format(dep_name, name), cpp_info_flags, 'defines') -}}
        {{- define_variable_value_safe("CONAN_CFLAGS_{}_{}".format(dep_name, name), cpp_info_flags, 'cflags') -}}
        {{- define_variable_value_safe("CONAN_CXXFLAGS_{}_{}".format(dep_name, name), cpp_info_flags, 'cxxflags') -}}
        {{- define_variable_value_safe("CONAN_SHAREDLINKFLAGS_{}_{}".format(dep_name, name), cpp_info_flags, 'sharedlinkflags') -}}
        {{- define_variable_value_safe("CONAN_EXELINKFLAGS_{}_{}".format(dep_name, name), cpp_info_flags, 'exelinkflags') -}}
        {{- define_variable_value_safe("CONAN_FRAMEWORKS_{}_{}".format(dep_name, name), cpp_info_flags, 'frameworks') -}}
        {{- define_variable_value_safe("CONAN_REQUIRES_{}_{}".format(dep_name, name), cpp_info_flags, 'requires') -}}
        {{- define_variable_value_safe("CONAN_SYSTEM_LIBS_{}_{}".format(dep_name, name), cpp_info_flags, 'system_libs') -}}
        {{- define_multiple_variable_value("CONAN_PROPERTY_{}_{}".format(dep_name, name), properties) -}}
        """)

    def __init__(self, dependency, component_name: str, dirs: dict, flags: dict, output):
        """
        :param dependency: The dependency object that owns the component
        :param component_name: component raw name e.g. poco::poco_json
        :param dirs: The component cpp_info folders
        :param flags: The component cpp_info variables
        """
        self._dep = dependency
        self._name = component_name
        self._dirs = dirs or {}
        self._flags = flags or {}
        self._output = output

    def content(self) -> str:
        """
        Format template and generate Makefile component
        """
        context = {
            "dep": self._dep,
            "comp_name": self._name,
            "dep_name": _makefy(self._dep.ref.name),
            "name": _makefy(self._name),
            "cpp_info_dirs": self._dirs,
            "cpp_info_flags": self._flags,
            "properties": _makefy_properties(_filter_properties(self._dep.cpp_info.components[self._name]._properties, self._output)),
        }
        template = Template(_jinja_format_list_values() + self.template, trim_blocks=True,
                            lstrip_blocks=True, undefined=StrictUndefined)
        return template.render(context)


class DepContentGenerator:
    """
    Generates Makefile content for a dependency
    """

    template = textwrap.dedent("""\

        # {{ dep.ref }}{% if not req.direct %} (indirect dependency){% endif +%}

        CONAN_NAME_{{ name }} = {{ dep.ref.name }}
        CONAN_VERSION_{{ name }} = {{ dep.ref.version }}
        CONAN_REFERENCE_{{ name }} = {{ dep.ref }}

        CONAN_ROOT_{{ name }} = {{ root }}

        {{  define_variable_value("CONAN_SYSROOT_{}".format(name), sysroot) -}}
        {{- define_variable_value_safe("CONAN_INCLUDE_DIRS_{}".format(name), cpp_info_dirs, 'include_dirs') -}}
        {{- define_variable_value_safe("CONAN_LIB_DIRS_{}".format(name), cpp_info_dirs, 'lib_dirs') -}}
        {{- define_variable_value_safe("CONAN_BIN_DIRS_{}".format(name), cpp_info_dirs, 'bin_dirs') -}}
        {{- define_variable_value_safe("CONAN_SRC_DIRS_{}".format(name), cpp_info_dirs, 'src_dirs') -}}
        {{- define_variable_value_safe("CONAN_BUILD_DIRS_{}".format(name), cpp_info_dirs, 'build_dirs') -}}
        {{- define_variable_value_safe("CONAN_RES_DIRS_{}".format(name), cpp_info_dirs, 'res_dirs') -}}
        {{- define_variable_value_safe("CONAN_FRAMEWORK_DIRS_{}".format(name), cpp_info_dirs, 'framework_dirs') -}}
        {{- define_variable_value_safe("CONAN_OBJECTS_{}".format(name), cpp_info_flags, 'objects') -}}
        {{- define_variable_value_safe("CONAN_LIBS_{}".format(name), cpp_info_flags, 'libs') -}}
        {{- define_variable_value_safe("CONAN_DEFINES_{}".format(name), cpp_info_flags, 'defines') -}}
        {{- define_variable_value_safe("CONAN_CFLAGS_{}".format(name), cpp_info_flags, 'cflags') -}}
        {{- define_variable_value_safe("CONAN_CXXFLAGS_{}".format(name), cpp_info_flags, 'cxxflags') -}}
        {{- define_variable_value_safe("CONAN_SHAREDLINKFLAGS_{}".format(name), cpp_info_flags, 'sharedlinkflags') -}}
        {{- define_variable_value_safe("CONAN_EXELINKFLAGS_{}".format(name), cpp_info_flags, 'exelinkflags') -}}
        {{- define_variable_value_safe("CONAN_FRAMEWORKS_{}".format(name), cpp_info_flags, 'frameworks') -}}
        {{- define_variable_value_safe("CONAN_REQUIRES_{}".format(name), cpp_info_flags, 'requires') -}}
        {{- define_variable_value_safe("CONAN_SYSTEM_LIBS_{}".format(name), cpp_info_flags, 'system_libs') -}}
        {{- define_variable_value("CONAN_COMPONENTS_{}".format(name), components) -}}
        {{- define_multiple_variable_value("CONAN_PROPERTY_{}".format(name), properties) -}}
        """)

    def __init__(self, dependency, require, root: str, sysroot, dirs: dict, flags: dict, output):
        self._dep = dependency
        self._req = require
        self._root = root
        self._sysroot = sysroot
        self._dirs = dirs or {}
        self._flags = flags or {}
        self._output = output

    def content(self) -> str:
        """
        Parse dependency variables and generate its Makefile content
        """
        context = {
            "dep": self._dep,
            "req": self._req,
            "name": _makefy(self._dep.ref.name),
            "root": self._root,
            "sysroot": self._sysroot,
            "components": list(self._dep.cpp_info.get_sorted_components().keys()),
            "cpp_info_dirs": self._dirs,
            "cpp_info_flags": self._flags,
            "properties": _makefy_properties(_filter_properties(self._dep.cpp_info._properties, self._output)),
        }
        template = Template(_jinja_format_list_values() + self.template, trim_blocks=True,
                            lstrip_blocks=True, undefined=StrictUndefined)
        return template.render(context)


class DepComponentGenerator:
    """
    Generates Makefile content for a dependency component
    """

    def __init__(self, dependency, makeinfo: MakeInfo, component_name: str, component, root: str, output):
        """
        :param dependency: The dependency object that owns the component
        :param makeinfo: Makeinfo to store component variables
        :param component_name: The component raw name e.g. poco::poco_json
        :param component: The component object to obtain cpp_info variables
        :param root: The dependency root folder
        """
        self._dep = dependency
        self._name = component_name
        self._comp = component
        self._root = root
        self._makeinfo = makeinfo
        self._output = output

    def _get_component_dirs(self) -> dict:
        """
        List regular directories from cpp_info and format them to be used in the makefile
        :return: A dictionary with regular folder name and its formatted path
        """
        dirs = {}
        for var, flag in _common_cppinfo_dirs().items():
            cppinfo_value = getattr(self._comp, var)
            formatted_dirs = _get_formatted_dirs(cppinfo_value, self._root, _makefy(self._name))
            if formatted_dirs:
                self._makeinfo.dirs_append(var)
                var = var.replace("dirs", "_dirs")
                formatted_dirs = self._rootify(self._root, self._dep.ref.name, cppinfo_value)
                dirs[var] = [_conan_prefix_flag(flag) + it for it in formatted_dirs]
        return dirs

    @staticmethod
    def _rootify(root: str, root_id: str, path_list: list) -> list:
        """
        Replaces component folder path by its root node folder path in case they match
        :param root: root folder path for component's father
        :param root_id: component's dependency name
        :param path_list: folder list available in the component
        :return: A formatted folder list, solving root folder path as prefix
        """
        root_len = len(root)
        root_with_sep = root + os.sep
        root_var_ref = f"$(CONAN_ROOT_{_makefy(root_id)})"
        return [root_var_ref + path[root_len:].replace("\\", "/") if path.startswith(root_with_sep)
                else path for path in path_list]

    def _get_component_flags(self) -> dict:
        """
        List common variables from cpp_info and format them to be used in the makefile
        :return: A dictionary with regular flag/variable name and its formatted value with prefix
        """
        flags = {}
        for var, prefix_var in _common_cppinfo_variables().items():
            cppinfo_value = getattr(self._comp, var)
            if not cppinfo_value:
                continue
            if "flags" in var:
                cppinfo_value = [var.replace('"', '\\"') for var in cppinfo_value]
            if cppinfo_value:
                flags[var] = [_conan_prefix_flag(prefix_var) + it for it in cppinfo_value]
                self._makeinfo.flags_append(var)
        return flags

    def generate(self) -> str:
        """
        Process component cpp_info variables and generate its Makefile content
        :return: Component Makefile content
        """
        dirs = self._get_component_dirs()
        flags = self._get_component_flags()
        comp_content_gen = DepComponentContentGenerator(self._dep, self._name, dirs, flags, self._output)
        comp_content = comp_content_gen.content()
        return comp_content


class DepGenerator:
    """
    Process a dependency cpp_info variables and generate its Makefile content
    """

    def __init__(self, dependency, require, output):
        self._dep = dependency
        self._req = require
        self._info = MakeInfo(self._dep.ref.name, [], [])
        self._output = output

    @property
    def makeinfo(self) -> MakeInfo:
        """
        :return: Dependency folder and flags
        """
        return self._info

    def _get_dependency_dirs(self, root: str, dependency) -> dict:
        """
        List regular directories from cpp_info and format them to be used in the makefile
        :param root: Package root folder
        :param dependency: Dependency object
        :return: A dictionary with regular folder name and its formatted path
        """
        dirs = {}
        for var, prefix in _common_cppinfo_dirs().items():
            cppinfo_value = getattr(dependency.cpp_info, var)
            if not cppinfo_value:  # The root value is not defined, there might be components
                cppinfo_value = [f"$(CONAN_{var.replace('dirs', '_dirs').upper()}_{_makefy(dependency.ref.name)}_{_makefy(name)})"
                                 for name, obj in dependency.cpp_info.components.items() if getattr(obj, var.lower())]
                prefix = ""
            formatted_dirs = _get_formatted_dirs(cppinfo_value, root, _makefy(dependency.ref.name))
            if formatted_dirs:
                self._info.dirs_append(var)
                var = var.replace("dirs", "_dirs")
                dirs[var] = [_conan_prefix_flag(prefix) + it for it in formatted_dirs]
        return dirs

    def _get_dependency_flags(self, dependency) -> dict:
        """
        List common variables from cpp_info and format them to be used in the makefile
        :param dependency: Dependency object
        """
        flags = {}
        for var, prefix_var in _common_cppinfo_variables().items():
            cppinfo_value = getattr(dependency.cpp_info, var)
            # Use component cpp_info info when does not provide any value
            if not cppinfo_value:
                cppinfo_value = [f"$(CONAN_{var.upper()}_{_makefy(dependency.ref.name)}_{_makefy(name)})" for name, obj in dependency.cpp_info.components.items() if getattr(obj, var.lower())]
                # avoid repeating same prefix twice
                prefix_var = ""
            if "flags" in var:
                cppinfo_value = [var.replace('"', '\\"') for var in cppinfo_value]
            if cppinfo_value:
                self._info.flags_append(var)
                flags[var] = [_conan_prefix_flag(prefix_var) + it for it in cppinfo_value]
        return flags

    def _get_sysroot(self, root: str) -> list:
        """
        Get the sysroot of the dependency. Sysroot is a list of directories, or a single directory
        """
        sysroot = self._dep.cpp_info.sysroot if isinstance(self._dep.cpp_info.sysroot, list) else [self._dep.cpp_info.sysroot]
        # sysroot may return ['']
        if not sysroot or not sysroot[0]:
            return []
        return _get_formatted_dirs(sysroot, root, _makefy(self._dep.ref.name)) if sysroot and sysroot[0] else None

    def _get_root_folder(self):
        """
        Get the root folder of the dependency
        """
        root = self._dep.recipe_folder if self._dep.package_folder is None else self._dep.package_folder
        return root.replace("\\", "/")

    def generate(self) -> str:
        """
        Process dependency folders and flags to generate its Makefile content. Plus, execute same
        steps for each component
        """
        root = self._get_root_folder()
        sysroot = self._get_sysroot(root)
        dirs = self._get_dependency_dirs(root, self._dep)
        flags = self._get_dependency_flags(self._dep)
        dep_content_gen = DepContentGenerator(self._dep, self._req, root, sysroot, dirs, flags, self._output)
        content = dep_content_gen.content()

        for comp_name, comp in self._dep.cpp_info.get_sorted_components().items():
            component_gen = DepComponentGenerator(self._dep, self._info, comp_name, comp, root, self._output)
            content += component_gen.generate()

        return content


class MakeDeps:
    """
    Generates a Makefile with the variables needed to build a project with the specified.
    """

    _title = "# This Makefile has been generated by Conan. DO NOT EDIT!\n"

    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile

    def generate(self) -> None:
        """
        Collects all dependencies and components, then, generating a Makefile
        """
        check_duplicated_generator(self, self._conanfile)

        host_req = self._conanfile.dependencies.host
        test_req = self._conanfile.dependencies.test

        content_buffer = f"{self._title}\n"
        deps_buffer = ""

        # Filter the build_requires not activated for any requirement
        dependencies = list(host_req.items()) + list(test_req.items())

        make_infos = []

        for require, dep in dependencies:
            output = ConanOutput(scope=f"{self._conanfile} MakeDeps: {dep}:")
            dep_gen = DepGenerator(dep, require, output)
            make_infos.append(dep_gen.makeinfo)
            deps_buffer += dep_gen.generate()

        glob_gen = GlobalGenerator(self._conanfile, make_infos)
        content_buffer += glob_gen.deps_generate() + deps_buffer + glob_gen.generate()

        save(self._conanfile, CONAN_MAKEFILE_FILENAME, content_buffer)
        self._conanfile.output.info(f"Generated {CONAN_MAKEFILE_FILENAME}")
