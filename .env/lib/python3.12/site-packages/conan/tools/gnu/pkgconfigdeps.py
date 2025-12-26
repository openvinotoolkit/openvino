import os
import re
import textwrap

from jinja2 import Template, StrictUndefined

from conan.errors import ConanException
from conan.internal import check_duplicated_generator
from conan.internal.model.dependencies import get_transitive_requires
from conan.internal.util.files import save


class _PCFilesDeps:

    template = textwrap.dedent("""\
        {% for k, v in pc_variables.items() %}
        {{ "{}={}".format(k, v) }}
        {% endfor %}

        Name: {{ name }}
        Description: {{ description }}
        Version: {{ version }}
        {% if libflags %}
        Libs: {{ libflags }}
        {% endif %}
        {% if cflags %}
        Cflags: {{ cflags }}
        {% endif %}
        {% if requires|length %}
        Requires: {{ requires|join(' ') }}
        {% endif %}
    """)

    alias_template = textwrap.dedent("""\
        Name: {{name}}
        Description: Alias {{name}} for {{aliased}}
        Version: {{version}}
        Requires: {{aliased}}
    """)

    def __init__(self, pkgconfigdeps, dep, suffix=""):
        self._conanfile = pkgconfigdeps._conanfile  # noqa
        self._properties = pkgconfigdeps._properties  # noqa
        self._transitive_reqs = get_transitive_requires(self._conanfile, dep)
        self._dep = dep
        self._suffix = suffix

    def _get_aliases(self, dep, pkg_name=None, comp_ref_name=None):
        def _get_dep_aliases():
            pkg_aliases = self._get_property("pkg_config_aliases", dep, check_type=list)
            return pkg_aliases or []

        # TODO: LET'S DEPRECATE ALL THE ALIASES MECHANISM!!
        if pkg_name is None and comp_ref_name is None:
            return _get_dep_aliases()
        if comp_ref_name not in dep.cpp_info.components:
            # Either foo::foo might be referencing the root cpp_info
            if (dep.ref.name == comp_ref_name or
                # Or a "replace_require" is used and cpp_info.requires is the root one, e.g.,
                # zlib/*: zlib-ng/*, and self.cpp_info.requires = ["zlib::zlib"]
                (dep.ref.name != pkg_name and pkg_name == comp_ref_name)):
                return _get_dep_aliases()
            raise ConanException("Component '{name}::{cname}' not found in '{name}' "
                                 "package requirement".format(name=dep.ref.name,
                                                              cname=comp_ref_name))
        comp_aliases = self._get_property("pkg_config_aliases", dep, comp_ref_name, check_type=list)
        return comp_aliases or []

    def _get_name(self, dep, pkg_name=None, comp_ref_name=None):
        def _get_dep_name():
            dep_name = self._get_property("pkg_config_name", dep) or dep.ref.name
            return f"{dep_name}{self._suffix}"

        if pkg_name is None and comp_ref_name is None:
            return _get_dep_name()
        if comp_ref_name not in dep.cpp_info.components:
            # Either foo::foo might be referencing the root cpp_info
            if (dep.ref.name == comp_ref_name or
                # Or a "replace_require" is used and cpp_info.requires is the root one, e.g.,
                # zlib/*: zlib-ng/*, and self.cpp_info.requires = ["zlib::zlib"]
                (dep.ref.name != pkg_name and pkg_name == comp_ref_name)):
                return _get_dep_name()
            raise ConanException("Component '{name}::{cname}' not found in '{name}' "
                                 "package requirement".format(name=dep.ref.name,
                                                              cname=comp_ref_name))
        comp_name = self._get_property("pkg_config_name", dep, comp_ref_name)
        if comp_name:
            return f"{comp_name}{self._suffix}"
        else:
            dep_name = _get_dep_name()
            # Creating a component name with namespace, e.g., dep-comp1
            return f"{dep_name}-{comp_ref_name}"

    def _get_property(self, prop, dep, comp_name=None, check_type=None):
        dep_name = dep.ref.name
        dep_comp = f"{str(dep_name)}::{comp_name}" if comp_name else f"{str(dep_name)}"
        try:
            value = self._properties[f"{dep_comp}{self._suffix}"][prop]
            if check_type is not None and not isinstance(value, check_type):
                raise ConanException(
                    f'The expected type for {prop} is "{check_type.__name__}", but "{type(value).__name__}" was found')
            return value
        except KeyError:
            return dep.cpp_info.get_property(prop, check_type=check_type) if not comp_name \
                else dep.cpp_info.components[comp_name].get_property(prop, check_type=check_type)

    def _get_pc_variables(self, dep, cpp_info, custom_content=None):
        """
        Get all the freeform variables defined by Conan and
        users (through ``pkg_config_custom_content``). This last ones will override the
        Conan defined variables.
        """
        def apply_custom_content():
            if isinstance(custom_content, dict):
                pc_variables.update(custom_content)
            elif custom_content:  # Legacy: custom content is string
                pc_variable_pattern = re.compile("^(.*)=(.*)")
                for line in custom_content.splitlines():
                    match = pc_variable_pattern.match(line)
                    if match:
                        key, value = match.group(1).strip(), match.group(2).strip()
                        pc_variables[key] = value

        # If editable, package_folder can be None
        prefix_path = (dep.recipe_folder if dep.package_folder is None
                                   else dep.package_folder).replace("\\", "/")
        pc_variables = {"prefix": prefix_path}
        # Already formatted directories
        pc_variables.update(self._get_formatted_dirs("libdir", cpp_info.libdirs, prefix_path))
        pc_variables.update(self._get_formatted_dirs("includedir", cpp_info.includedirs, prefix_path))
        pc_variables.update(self._get_formatted_dirs("bindir", cpp_info.bindirs, prefix_path))
        # Get the custom content introduced by user and sanitize it
        apply_custom_content()
        return pc_variables

    @staticmethod
    def _get_formatted_dirs(folder_name, folders, prefix_path_):
        ret = {}
        for i, directory in enumerate(folders):
            directory = os.path.normpath(directory).replace("\\", "/")
            if directory.startswith(prefix_path_):
                prefix = "${prefix}/"
                directory = os.path.relpath(directory, prefix_path_).replace("\\", "/")
            else:
                prefix = "" if os.path.isabs(directory) else "${prefix}/"
            suffix = str(i) if i else ""
            var_name = f"{folder_name}{suffix}"
            ret[var_name] = f"{prefix}{directory}"
        return ret

    def _get_framework_flags(self, cpp_info):
        # FIXME: GnuDepsFlags used only here. Let's adapt the code and remove this dependency.
        #        self._conanfile is also used only here.
        from conan.tools.gnu.gnudeps_flags import GnuDepsFlags
        gnudeps_flags = GnuDepsFlags(self._conanfile, cpp_info)
        return gnudeps_flags.frameworks + gnudeps_flags.framework_paths

    def _get_lib_flags(self, libdirvars, cpp_info):
        framework_flags = self._get_framework_flags(cpp_info)
        libdirsflags = ['-L"${%s}"' % d for d in libdirvars]
        system_libs = ["-l%s" % li for li in (cpp_info.libs + cpp_info.system_libs)]
        shared_flags = cpp_info.sharedlinkflags + cpp_info.exelinkflags
        return " ".join(libdirsflags + system_libs + shared_flags + framework_flags)

    def _get_cflags(self, includedirvars, cpp_info):
        includedirsflags = ['-I"${%s}"' % d for d in includedirvars]
        cxxflags = [var.replace('"', '\\"') for var in cpp_info.cxxflags]
        cflags = [var.replace('"', '\\"') for var in cpp_info.cflags]
        defines = ["-D%s" % var.replace('"', '\\"') for var in cpp_info.defines]
        return " ".join(includedirsflags + cxxflags + cflags + defines)

    def _get_component_requirement_names(self, cpp_info):
        """
        Get all the pkg-config valid names from the requirements ones given a CppInfo object.

        For instance, those requirements could be coming from:

        ```python
        from conan import ConanFile
        class PkgConfigConan(ConanFile):
            requires = "other/1.0"

            def package_info(self):
                self.cpp_info.requires = ["other::cmp1"]

            # Or:

            def package_info(self):
                self.cpp_info.components["cmp"].requires = ["other::cmp1"]
        ```
        """
        dep_ref_name = self._dep.ref.name
        ret = []
        for req in cpp_info.requires:
            pkg_ref_name, comp_ref_name = req.split("::") if "::" in req else (dep_ref_name, req)
            # For instance, dep == "hello/1.0" and req == "other::cmp1" -> hello != other
            if dep_ref_name != pkg_ref_name:
                try:
                    req_conanfile = self._transitive_reqs[pkg_ref_name]
                except KeyError:
                    continue  # If the dependency is not in the transitive, might be skipped
            else:  # For instance, dep == "hello/1.0" and req == "hello::cmp1" -> hello == hello
                req_conanfile = self._dep
            comp_name = self._get_name(req_conanfile, pkg_ref_name, comp_ref_name)
            if comp_name not in ret:
                ret.append(comp_name)
        return ret

    def items(self):
        """
        Get all the PC files and contents for any dependency:

        * If the given dependency does not have components:
            The PC file will be the dependency one.

        * If the given dependency has components:
            The PC files will be saved in this order:
                1- Package components.
                2- Root component.

            Note: If the root-package PC name matches with any other of the components one, the first one
            is not going to be created. Components have more priority than root package.

        * Apart from those PC files, if there are any aliases declared, they will be created too.
        """
        pc_files = {}
        pc_alias_files = {}
        pkg_name = self._get_name(self._dep)
        # First, let's load all the components PC files
        # Loop through all the package's components
        for comp_ref_name, comp_cpp_info in self._dep.cpp_info.get_sorted_components().items():
            # At first, let's check if we have defined some components requires, e.g., "dep::cmp1"
            comp_requires = self._get_component_requirement_names(comp_cpp_info)
            comp_name = self._get_name(self._dep, pkg_name, comp_ref_name)
            version = (self._get_property("component_version", self._dep, comp_ref_name) or
                       self._get_property("system_package_version", self._dep, comp_ref_name) or
                       self._dep.ref.version)
            custom_content = self._get_property("pkg_config_custom_content", self._dep, comp_ref_name)
            pc_variables = self._get_pc_variables(self._dep, comp_cpp_info, custom_content)
            pc_context = {
                "name": comp_name,
                "description": f"Conan component: {comp_name}",
                "version": version,
                "requires": comp_requires,
                "pc_variables": pc_variables,
                "cflags": self._get_cflags([d for d in pc_variables if d.startswith("includedir")],
                                           comp_cpp_info),
                "libflags": self._get_lib_flags([d for d in pc_variables if d.startswith("libdir")],
                                                comp_cpp_info)
            }
            pc_files[comp_name] = self._get_pc_content(pc_context)
            # Aliases
            for alias in self._get_aliases(self._dep, pkg_name, comp_ref_name):
                pc_alias_files[alias] = self._get_alias_pc_content({
                    "name": alias,
                    "version": version,
                    "aliased": comp_name
                })
        # Second, let's load the root package's PC file ONLY
        # if it does not already exist in components one
        # Issue related: https://github.com/conan-io/conan/issues/10341
        should_skip_main = self._get_property("pkg_config_name", self._dep) == "none"
        if pkg_name not in pc_files and not should_skip_main:
            cpp_info = self._dep.cpp_info
            # At first, let's check if we have defined some global requires, e.g., "other::cmp1"
            # Note: If DEP has components, they'll be the requirements == pc_files.keys()
            requires = list(pc_files.keys()) or self._get_component_requirement_names(cpp_info)
            # If we have found some component requirements it would be enough
            if not requires:
                # If no requires were found, let's try to get all the direct visible dependencies,
                # e.g., requires = "other_pkg/1.0"
                requires = [self._get_name(req) for req in self._transitive_reqs.values()]
            version = (self._get_property("system_package_version", self._dep)
                       or self._dep.ref.version)
            custom_content = self._get_property("pkg_config_custom_content", self._dep)
            pc_variables = self._get_pc_variables(self._dep, cpp_info, custom_content)
            pc_context = {
                "name": pkg_name,
                "description": f"Conan package: {pkg_name}",
                "version": version,
                "requires": requires,
                "pc_variables": pc_variables,
                "cflags": self._get_cflags([d for d in pc_variables if d.startswith("includedir")],
                                           cpp_info),
                "libflags": self._get_lib_flags([d for d in pc_variables if d.startswith("libdir")],
                                                cpp_info)
            }
            pc_files[pkg_name] = self._get_pc_content(pc_context)
            # Aliases
            for alias in self._get_aliases(self._dep):
                pc_alias_files[alias] = self._get_alias_pc_content({
                    "name": alias,
                    "version": version,
                    "aliased": pkg_name
                })
        # Adding the aliases
        pc_files.update(pc_alias_files)
        return pc_files.items()

    def _get_pc_content(self, context):
        template = Template(self.template, trim_blocks=True, lstrip_blocks=True,
                            undefined=StrictUndefined)
        return template.render(context)

    def _get_alias_pc_content(self, context):
        template = Template(self.alias_template, trim_blocks=True, lstrip_blocks=True,
                            undefined=StrictUndefined, keep_trailing_newline=True)
        return template.render(context)


class PkgConfigDeps:

    def __init__(self, conanfile):
        self._conanfile = conanfile
        # Activate the build *.pc files for the specified libraries
        self.build_context_activated = []
        # If specified, the files/requires/names for the build context will be renamed appending
        # a suffix. It is necessary in case of same require and build_require and will cause an error
        # DEPRECATED: consumers should use build_context_folder instead
        # FIXME: Conan 3.x: Remove build_context_suffix attribute
        self.build_context_suffix = {}
        # By default, the "[generators_folder]/build" folder will save all the *.pc files activated
        # in the build_context_activated list.
        # Notice that if the `build_context_suffix` attr is defined, the `build_context_folder` one
        # will have no effect.
        # Issue: https://github.com/conan-io/conan/issues/12342
        # Issue: https://github.com/conan-io/conan/issues/14935
        # FIXME: Conan 3.x: build_context_folder should be "build" by default
        self.build_context_folder = None  # Keeping backward-compatibility
        self._properties = {}

    def _get_dependencies(self):
        # Get all the dependencies
        host_req = self._conanfile.dependencies.host
        build_req = self._conanfile.dependencies.build  # tool_requires
        test_req = self._conanfile.dependencies.test
        # If self.build_context_suffix is not defined, the build requires will be saved
        # in the self.build_context_folder
        # FIXME: Conan 3.x: Remove build_context_suffix attribute and the validation function
        if self.build_context_folder is None:  # Legacy flow
            if self.build_context_suffix:
                # deprecation warning
                self._conanfile.output.warning(
                    "PkgConfigDeps.build_context_suffix attribute has been "
                    "deprecated. Use PkgConfigDeps.build_context_folder instead."
                )
            # Check if it exists both as require and as build require without a suffix
            activated_br = {r.ref.name for r in build_req.values()
                            if r.ref.name in self.build_context_activated}
            common_names = {r.ref.name for r in host_req.values()}.intersection(activated_br)
            without_suffixes = [common_name for common_name in common_names
                                if not self.build_context_suffix.get(common_name)]
            if without_suffixes:
                raise ConanException(
                    f"The packages {without_suffixes} exist both as 'require' and as"
                    f" 'build require'. You need to specify a suffix using the "
                    f"'build_context_suffix' attribute at the PkgConfigDeps generator.")
        elif self.build_context_folder is not None and self.build_context_suffix:
            raise ConanException(
                "It's not allowed to define both PkgConfigDeps.build_context_folder "
                "and PkgConfigDeps.build_context_suffix (deprecated).")

        for require, dep in list(host_req.items()) + list(build_req.items()) + list(test_req.items()):
            # Filter the build_requires not activated with PkgConfigDeps.build_context_activated
            if require.build and dep.ref.name not in self.build_context_activated:
                continue
            yield require, dep

    def generate(self):
        """
        Save all the `*.pc` files
        """
        def _pc_file_name(name_, is_build_context=False, has_suffix=False):
            # If no suffix is defined, we can save the *.pc file in the build_context_folder
            build = is_build_context and self.build_context_folder and not has_suffix
            # Issue: https://github.com/conan-io/conan/issues/12342
            # Issue: https://github.com/conan-io/conan/issues/14935
            return f"{self.build_context_folder}/{name_}.pc" if build else f"{name_}.pc"

        check_duplicated_generator(self, self._conanfile)
        for require, dep in self._get_dependencies():
            suffix = self.build_context_suffix.get(require.ref.name, "") if require.build else ""
            # Save all the *.pc files and their contents
            for name, content in _PCFilesDeps(self, dep, suffix=suffix).items():
                pc_name = _pc_file_name(name, is_build_context=require.build,
                                        has_suffix=bool(suffix))
                save(pc_name, content)

    def set_property(self, dep, prop, value):
        """
        Using this method you can overwrite the :ref:`property<PkgConfigDeps Properties>` values set by
        the Conan recipes from the consumer. This can be done for `pkg_config_name`,
        `pkg_config_aliases` and `pkg_config_custom_content` properties.

        :param dep: Name of the dependency to set the :ref:`property<PkgConfigDeps Properties>`. For
         components use the syntax: ``dep_name::component_name``.
        :param prop: Name of the :ref:`property<PkgConfigDeps Properties>`.
        :param value: Value of the property. Use ``None`` to invalidate any value set by the
         upstream recipe.
        """
        self._properties.setdefault(dep, {}).update({prop: value})
