import os
import re
import textwrap

from jinja2 import Template, StrictUndefined

from conan.internal import check_duplicated_generator
from conan.internal.model.dependencies import get_transitive_requires
from conan.internal.model.pkg_type import PackageType
from conan.internal.util.files import save


def _relativize_path(path, start_folder):
    """
    Returns a relative path with regard to the given folder.

    :param path: absolute or relative path.
    :param start_folder: folder to start relative to.
    :return: Unix-like path relative if matches to the given pattern.
             Otherwise, it returns the original path.
    """
    if not path or not start_folder:
        return path
    path_ = path.replace("\\", "/").replace("/./", "/")
    pattern_ = start_folder.replace("\\", "/").replace("/./", "/")
    match = re.match(pattern_, path_)
    if match:
        matching = match[0]
        if path_.startswith(matching):
            path_ = path_.replace(matching, "").strip("/")
            return path_.strip("./") or "./"
    return path


class _BazelDepBuildGenerator:
    """
    This class creates the BUILD.bazel for each dependency where it's declared all the
    necessary information to load the libraries
    """
    # If both files exist, BUILD.bazel takes precedence over BUILD
    # https://bazel.build/concepts/build-files
    dep_build_filename = "BUILD.bazel"
    dep_build_template = textwrap.dedent("""\
    {% macro cc_import_macro(libs) %}
    {% for lib_info in libs %}
    cc_import(
        name = "{{ lib_info['name'] }}_precompiled",
        {% if lib_info['is_shared'] %}
        shared_library = "{{ lib_info['lib_path'] }}",
        {% else %}
        static_library = "{{ lib_info['lib_path'] }}",
        {% endif %}
        {% if lib_info['import_lib_path'] %}
        interface_library = "{{ lib_info['import_lib_path'] }}",
        {% endif %}
        {% if lib_info["linkopts"] %}
        linkopts = [
            {% for linkopt in lib_info["linkopts"] %}
            {{ linkopt }},
            {% endfor %}
        ],
        {% endif %}
    )
    {% endfor %}
    {% endmacro %}
    {% macro cc_library_macro(obj) %}
    cc_library(
        name = "{{ obj["name"] }}",
        {% if obj["headers"] %}
        hdrs = glob([
            {% for header in obj["headers"] %}
            {{ header }},
            {% endfor %}
        ],
        allow_empty = True
        ),
        {% endif %}
        {% if obj["includes"] %}
        includes = [
            {% for include in obj["includes"] %}
            {{ include }},
            {% endfor %}
        ],
        {% endif %}
        {% if obj["defines"] %}
        defines = [
            {% for define in obj["defines"] %}
            {{ define }},
            {% endfor %}
        ],
        {% endif %}
        {% if obj["linkopts"] %}
        linkopts = [
            {% for linkopt in obj["linkopts"] %}
            {{ linkopt }},
            {% endfor %}
        ],
        {% endif %}
        {% if obj["copts"] %}
        copts = [
            {% for copt in obj["copts"] %}
            {{ copt }},
            {% endfor %}
        ],
        {% endif %}
        visibility = ["//visibility:public"],
        {% if obj["libs"] or obj["dependencies"] or obj.get("component_names", []) %}
        deps = [
            # do not sort
            {% for lib in obj["libs"] %}
            ":{{ lib.name }}_precompiled",
            {% endfor %}
            {% for name in obj.get("component_names", []) %}
            ":{{ name }}",
            {% endfor %}
            {% for dep in obj["dependencies"] %}
            "{{ dep }}",
            {% endfor %}
        ],
        {% endif %}
    )
    {% endmacro %}
    {% macro filegroup_bindirs_macro(obj) %}
    {% if obj["bindirs"] %}
    filegroup(
        name = "{{ obj["name"] }}_binaries",
        srcs = glob([
            {% for bindir in obj["bindirs"] %}
            "{{ bindir }}/**",
            {% endfor %}
        ],
        allow_empty = True
        ),
        visibility = ["//visibility:public"],
    )
    {% endif %}
    {% endmacro %}
    # Components precompiled libs
    {% for component in components %}
    {{ cc_import_macro(component["libs"]) }}
    {% endfor %}
    # Root package precompiled libs
    {{ cc_import_macro(root["libs"]) }}
    # Components libraries declaration
    {% for component in components %}
    {{ cc_library_macro(component) }}
    {% endfor %}
    # Package library declaration
    {{ cc_library_macro(root) }}
    # Filegroup library declaration
    {{ filegroup_bindirs_macro(root) }}
    """)

    def __init__(self, conanfile, dep, require):
        self._conanfile = conanfile
        self._dep = dep
        self._is_build_require = require.build
        self._transitive_reqs = get_transitive_requires(self._conanfile, dep)

    @property
    def _build_file_path(self):
        """
        Returns the absolute path to the BUILD file created by Conan
        """
        folder = os.path.join(self._get_repository_name(self._dep), self.dep_build_filename)
        return folder.replace("\\", "/")

    @property
    def _absolute_build_file_path(self):
        """
        Returns the absolute path to the BUILD file created by Conan
        """
        folder = os.path.join(self._conanfile.generators_folder, self._build_file_path)
        return folder.replace("\\", "/")

    @property
    def _package_folder(self):
        """
        Returns the package folder path
        """
        # If editable, package_folder can be None
        root_folder = self._dep.recipe_folder if self._dep.package_folder is None \
            else self._dep.package_folder
        return root_folder.replace("\\", "/")

    def _get_repository_name(self, dep):
        pkg_name = dep.cpp_info.get_property("bazel_repository_name") or dep.ref.name
        return f"build-{pkg_name}" if self._is_build_require else pkg_name

    def _get_target_name(self, dep):
        pkg_name = dep.cpp_info.get_property("bazel_target_name") or dep.ref.name
        return pkg_name

    def _get_component_name(self, dep, comp_ref_name):
        pkg_name = self._get_target_name(dep)
        if comp_ref_name not in dep.cpp_info.components:
            # foo::foo might be referencing the root cppinfo
            if dep.ref.name == comp_ref_name:
                return pkg_name
            return f"{pkg_name}-{comp_ref_name}"
        comp_name = dep.cpp_info.components[comp_ref_name].get_property("bazel_target_name")
        # If user did not set bazel_target_name, let's create a component name
        # with a namespace, e.g., dep-comp1
        return comp_name or f"{pkg_name}-{comp_ref_name}"

    def _get_headers(self, cpp_info):
        return ['"{}/**"'.format(_relativize_path(path, self._package_folder))
                for path in cpp_info.includedirs]

    def _get_bindirs(self, cpp_info):
        return [_relativize_path(bindir, self._package_folder) for bindir in cpp_info.bindirs]

    def _get_includes(self, cpp_info):
        return ['"{}"'.format(_relativize_path(path, self._package_folder))
                for path in cpp_info.includedirs]

    def _get_defines(self, cpp_info):
        return ['"{}"'.format(define.replace('"', '\\' * 3 + '"'))
                for define in cpp_info.defines]

    def _get_linkopts(self, cpp_info):
        os_build = self._dep.settings_build.get_safe("os")
        link_opt = '/DEFAULTLIB:{}' if os_build == "Windows" else '-l{}'
        system_libs = [link_opt.format(lib) for lib in cpp_info.system_libs]
        shared_flags = cpp_info.sharedlinkflags + cpp_info.exelinkflags
        frameworkdirs_flags = []
        framework_flags = []
        for frw in cpp_info.frameworks:
            framework_flags.extend(["-framework", frw])
        for frw_dir in cpp_info.frameworkdirs:
            frameworkdirs_flags.extend(["-F", frw_dir.replace("\\", "/")])
        return [f'"{flag}"' for flag in (system_libs + shared_flags + framework_flags + frameworkdirs_flags)]

    def _get_copts(self, cpp_info):
        # FIXME: long discussions between copts (-Iflag) vs includes in Bazel. Not sure yet
        # includedirsflags = ['"-I{}"'.format(_relativize_path(d, package_folder_path))
        #                     for d in cpp_info.includedirs]
        cxxflags = [var.replace('"', '\\"') for var in cpp_info.cxxflags]
        cflags = [var.replace('"', '\\"') for var in cpp_info.cflags]
        return [f'"{flag}"' for flag in (cxxflags + cflags)]

    def _get_component_requirement_names(self, cpp_info):
        """
        Get all the valid names from the requirements ones given a CppInfo object.

        For instance, those requirements could be coming from:

        ```python
        from conan import ConanFile
        class PkgConan(ConanFile):
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
            prefix = ":"  # Requirements declared in the same BUILD file
            # For instance, dep == "hello/1.0" and req == "other::cmp1" -> hello != other
            if dep_ref_name != pkg_ref_name:
                try:
                    req_conanfile = self._transitive_reqs[pkg_ref_name]
                    # Requirements declared in another dependency BUILD file
                    prefix = f"@{self._get_repository_name(req_conanfile)}//:"
                except KeyError:
                    continue  # If the dependency is not in the transitive, might be skipped
            else:  # For instance, dep == "hello/1.0" and req == "hello::cmp1" -> hello == hello
                req_conanfile = self._dep
            comp_name = self._get_component_name(req_conanfile, comp_ref_name)
            dep_name = f"{prefix}{comp_name}"
            if dep_name not in ret:
                ret.append(dep_name)
        return ret

    def _get_lib_info(self, cpp_info, deduced_cpp_info, component_name=None):

        def _lib_info(lib_name, virtual_cpp_info):
            info = {
                "name": lib_name,
                "is_shared": virtual_cpp_info.type == PackageType.SHARED,
                "lib_path": _relativize_path(virtual_cpp_info.location, self._package_folder),
                "import_lib_path": _relativize_path(virtual_cpp_info.link_location, self._package_folder),
                "linkopts": []
            }
            if info['is_shared'] and info["lib_path"] and not info["lib_path"].endswith(".dll"):
                # Issue: https://github.com/conan-io/conan/issues/19190
                # Issue: https://github.com/conan-io/conan/issues/19135
                # (UNIX) Adding the rpath flag as any application could link through the library
                # which points out a symlink, but that name does not appear in the library location
                info["linkopts"] = [f'"-Wl,-rpath,{libdir}"' for libdir in cpp_info.libdirs]
            return info

        libs = cpp_info.libs
        libs_info = []
        if libs:
            if len(libs) > 1:
                for lib_name in libs:
                    name = f'_{component_name}_{lib_name}' if component_name else f'_{lib_name}'
                    virtual_cpp_info = deduced_cpp_info.components[name]
                    libs_info.append(_lib_info(lib_name, virtual_cpp_info))
            else:
                lib_name = libs[0]
                virtual_cpp_info = deduced_cpp_info.components[component_name] if component_name else deduced_cpp_info
                libs_info.append(_lib_info(lib_name, virtual_cpp_info))
        return libs_info

    def _get_build_file_context(self):
        """
        Get the whole package information

        :return: `_BazelTargetInfo` object with the package information
        """
        build_content = {"components": [], "root": {}}
        component_names = []
        deduced_cpp_info = self._dep.cpp_info.deduce_full_cpp_info(self._dep)

        if self._dep.cpp_info.has_components:
            # Loop through all the package's components
            for comp_ref_name, cmp_cpp_info in self._dep.cpp_info.get_sorted_components().items():
                # At first, let's check if we have defined some components requires, e.g., "dep::cmp1"
                comp_requires_names = self._get_component_requirement_names(cmp_cpp_info)
                comp_name = self._get_component_name(self._dep, comp_ref_name)
                component_names.append(comp_name)
                build_content["components"].append({
                    "name": comp_name,
                    "libs": self._get_lib_info(cmp_cpp_info, deduced_cpp_info, component_name=comp_ref_name),
                    "bindirs": self._get_bindirs(cmp_cpp_info),
                    "headers": self._get_headers(cmp_cpp_info),
                    "includes": self._get_includes(cmp_cpp_info),
                    "defines": self._get_defines(cmp_cpp_info),
                    "linkopts": self._get_linkopts(cmp_cpp_info),
                    "copts": self._get_copts(cmp_cpp_info),
                    "dependencies": comp_requires_names,
                })

        pkg_name = self._get_target_name(self._dep)
        # At first, let's check if we have defined some global requires, e.g., "other::cmp1"
        requires = self._get_component_requirement_names(self._dep.cpp_info)
        # If we have found some component requires it would be enough
        if not requires:
            # If no requires were found, let's try to get all the direct dependencies,
            # e.g., requires = "other_pkg/1.0"
            requires = [
                f"@{self._get_repository_name(req)}//:{self._get_target_name(req)}"
                for req in self._transitive_reqs.values()
            ]
        cpp_info = self._dep.cpp_info
        build_content["root"] = {
            "name": pkg_name,
            "libs": self._get_lib_info(cpp_info, deduced_cpp_info),
            "bindirs": self._get_bindirs(cpp_info),
            "headers": self._get_headers(cpp_info),
            "includes": self._get_includes(cpp_info),
            "defines": self._get_defines(cpp_info),
            "linkopts": self._get_linkopts(cpp_info),
            "copts": self._get_copts(cpp_info),
            "dependencies": requires,
            "component_names": component_names
        }
        return build_content

    @property
    def dep_context(self):
        """
        Return the dependency context to fill later the conan_deps_module_extension.bzl and so on.
        """
        return {
            'repository_name': self._get_repository_name(self._dep),
            'package_folder': self._package_folder,
            'package_build_file_path': self._absolute_build_file_path,
        }

    def items(self):
        template = Template(self.dep_build_template, trim_blocks=True, lstrip_blocks=True,
                            undefined=StrictUndefined)
        content = template.render(self._get_build_file_context())
        return {self._build_file_path: content}.items()


class _BazelPathsGenerator:
    """
    Bazel 6.0 needs to know all the dependencies for its current project. So, the only way
    to do that is to tell the WORKSPACE file how to load all the Conan ones. This is the goal
    of the function created by this class, the ``load_conan_dependencies`` one.

    More information:
        * https://bazel.build/reference/be/workspace#new_local_repository

    Bazel >= 7.1 needs to know all the dependencies as well, but provided via the MODULE.bazel file.
    Therefor we provide a static repository rule to load the dependencies. This rule is used by a
    module extension, passing the package path and the BUILD file path to the repository rule.
    """
    repository_filename = "dependencies.bzl"
    modules_filename = "conan_deps_module_extension.bzl"
    repository_rules_filename = "conan_deps_repo_rules.bzl"
    repository_template = textwrap.dedent("""\
        # This Bazel module should be loaded by your WORKSPACE file.
        # Add these lines to your WORKSPACE one (assuming that you're using the "bazel_layout"):
        # load("@//conan:dependencies.bzl", "load_conan_dependencies")
        # load_conan_dependencies()

        def load_conan_dependencies():
        {% for dep_info in dependencies %}
            native.new_local_repository(
                name="{{dep_info['repository_name']}}",
                path="{{dep_info['package_folder']}}",
                build_file="{{dep_info['package_build_file_path']}}",
            )
        {% endfor %}
        """)
    module_template = textwrap.dedent("""\
        # This module provides a repo for each requires-dependency in your conanfile.
        # It's generated by the BazelDeps, and should be used in your Module.bazel file.
        load(":conan_deps_repo_rules.bzl", "conan_dependency_repo")

        def _load_dependencies_impl(mctx):
        {% for dep_info in dependencies %}
            conan_dependency_repo(
                name = "{{dep_info['repository_name']}}",
                package_path = "{{dep_info['package_folder']}}",
                build_file_path = "{{dep_info['package_build_file_path']}}",
            )
        {% endfor %}

            return mctx.extension_metadata(
                # It will only warn you if any direct
                # dependency is not imported by the 'use_repo' or even it is imported
                # but not created. Notice that root_module_direct_dev_deps can not be None as we
                # are giving 'all' value to root_module_direct_deps.
                # Fix the 'use_repo' calls by running 'bazel mod tidy'
                root_module_direct_deps = 'all',
                root_module_direct_dev_deps = [],

                # Prevent writing function content to lockfiles:
                # - https://bazel.build/rules/lib/builtins/module_ctx#extension_metadata
                # Important for remote build. Actually it's not reproducible, as local paths will
                # be different on different machines. But we assume that conan works correctly here.
                # IMPORTANT: Not compatible with bazel < 7.1
                reproducible = True,
            )

        conan_extension = module_extension(
            implementation = _load_dependencies_impl,
            os_dependent = True,
            arch_dependent = True,
        )
        """)
    repository_rules_content = textwrap.dedent("""\
        # This bazel repository rule is used to load Conan dependencies into the Bazel workspace.
        # It's used by a generated module file that provides information about the conan packages.
        # Each conan package is loaded into a bazel repository rule, with having the name of the
        # package. The whole method is based on symlinks to not copy the whole package into the
        # Bazel workspace, which is expensive.
        def _conan_dependency_repo(rctx):
            package_path = rctx.workspace_root.get_child(rctx.attr.package_path)

            child_packages = package_path.readdir()
            for child in child_packages:
                rctx.symlink(child, child.basename)

            rctx.symlink(rctx.attr.build_file_path, "BUILD.bazel")

        conan_dependency_repo = repository_rule(
            implementation = _conan_dependency_repo,
            attrs = {
                "package_path": attr.string(
                    mandatory = True,
                    doc = "The path to the Conan package in conan cache.",
                ),
                "build_file_path": attr.string(
                    mandatory = True,
                    doc = "The path to the BUILD file.",
                ),
            },
        )
        """)

    @classmethod
    def items(cls, dependencies_context):
        if not dependencies_context:
            return {}
        # Bazel 6.x, but it'll likely be dropped soon
        repository_template = Template(cls.repository_template, trim_blocks=True,
                                       lstrip_blocks=True,
                                       undefined=StrictUndefined)
        content_6x = repository_template.render(dependencies=dependencies_context)
        # Bazel 7.x files
        module_template = Template(cls.module_template, trim_blocks=True, lstrip_blocks=True,
                                   undefined=StrictUndefined)
        content = module_template.render(dependencies=dependencies_context)
        return {
            cls.repository_filename: content_6x,  # bazel 6.x compatible
            cls.modules_filename: content,
            cls.repository_rules_filename: cls.repository_rules_content,
            "BUILD.bazel": "# This is an empty BUILD file."  # Bazel needs this file in each subfolder
        }.items()


class BazelDeps:

    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile
        #: Activates the build context for the specified Conan package names.
        self.build_context_activated = []

    def _get_requirements(self, build_context_activated):
        """
        Simply save the activated requirements (host + build + test), and the deactivated ones
        """
        # All the requirements
        host_req = self._conanfile.dependencies.host
        build_req = self._conanfile.dependencies.direct_build  # tool_requires
        test_req = self._conanfile.dependencies.test

        for require, dep in list(host_req.items()) + list(build_req.items()) + list(
            test_req.items()):
            # Require is not used at the moment, but its information could be used,
            # and will be used in Conan 2.0
            # Filter the build_requires not activated with self.build_context_activated
            if require.build and dep.ref.name not in build_context_activated:
                continue
            yield require, dep

    def generate(self):
        """
        Generates all the targets <DEP>/BUILD.bazel files, a dependencies.bzl (for bazel<7), a
        conan_deps_repo_rules.bzl and a conan_deps_module_extension.bzl file (for bazel>=7.1) one in the
        build folder.

        In case of bazel < 7, it's important to highlight that the ``dependencies.bzl`` file should
        be loaded by your WORKSPACE Bazel file:

        .. code-block:: python

            load("@//[BUILD_FOLDER]:dependencies.bzl", "load_conan_dependencies")
            load_conan_dependencies()

        In case of bazel >= 7.1, the ``conan_deps_module_extension.bzl`` file should be loaded by your
        Module.bazel file, e.g. like this:

        .. code-block:: python

            load_conan_dependencies = use_extension(
                "//build:conan_deps_module_extension.bzl",
                "conan_extension"
            )
            use_repo(load_conan_dependencies, "dep-1", "dep-2", ...)
        """
        check_duplicated_generator(self, self._conanfile)
        dependencies_context = []
        for require, dep in self._get_requirements(self.build_context_activated):
            # Bazel info generator
            dep_build_generator = _BazelDepBuildGenerator(self._conanfile, dep, require)
            dependencies_context.append(dep_build_generator.dep_context)
            for name, content in dep_build_generator.items():
                save(name, content)

        for name, content in _BazelPathsGenerator.items(dependencies_context):
            save(name, content)
