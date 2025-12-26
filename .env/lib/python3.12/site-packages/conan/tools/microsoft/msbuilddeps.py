import fnmatch
import os
import re
import textwrap
from xml.dom import minidom

from jinja2 import Template

from conan.internal import check_duplicated_generator
from conan.errors import ConanException
from conan.internal.api.install.generators import relativize_path
from conan.internal.model.dependencies import get_transitive_requires
from conan.tools.microsoft.visual import msvc_platform_from_arch
from conan.internal.util.files import load, save

VALID_LIB_EXTENSIONS = (".so", ".lib", ".a", ".dylib", ".bc")


class MSBuildDeps:
    """
    MSBuildDeps class generator
    conandeps.props: unconditional import of all *direct* dependencies only
    """

    _vars_props = textwrap.dedent("""\
        <?xml version="1.0" encoding="utf-8"?>
        <Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
          <PropertyGroup Label="ConanVariables">
            <Conan{{name}}RootFolder>{{root_folder}}</Conan{{name}}RootFolder>
            <Conan{{name}}BinaryDirectories>{{bin_dirs}}</Conan{{name}}BinaryDirectories>
            {% if host_context %}
            <Conan{{name}}CompilerFlags>{{compiler_flags}}</Conan{{name}}CompilerFlags>
            <Conan{{name}}LinkerFlags>{{linker_flags}}</Conan{{name}}LinkerFlags>
            <Conan{{name}}PreprocessorDefinitions>{{definitions}}</Conan{{name}}PreprocessorDefinitions>
            <Conan{{name}}IncludeDirectories>{{include_dirs}}</Conan{{name}}IncludeDirectories>
            <Conan{{name}}ResourceDirectories>{{res_dirs}}</Conan{{name}}ResourceDirectories>
            <Conan{{name}}LibraryDirectories>{{lib_dirs}}</Conan{{name}}LibraryDirectories>
            <Conan{{name}}Libraries>{{libs}}</Conan{{name}}Libraries>
            <Conan{{name}}SystemLibs>{{system_libs}}</Conan{{name}}SystemLibs>
            {% endif %}
          </PropertyGroup>
        </Project>
        """)

    _conf_props = textwrap.dedent("""\
        <?xml version="1.0" encoding="utf-8"?>
        <Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
          <ImportGroup Label="PropertySheets">
            {% for dep in deps %}
            <Import Condition="'$(conan_{{dep}}_props_imported)' != 'True'" Project="conan_{{dep}}.props"/>
            {% endfor %}
          </ImportGroup>
          <ImportGroup Label="PropertySheets">
            <Import Project="{{vars_filename}}"/>
          </ImportGroup>
          {% if host_context %}
          <PropertyGroup>
            <ConanDebugPath>$(Conan{{name}}BinaryDirectories);$(ConanDebugPath)</ConanDebugPath>
            <LocalDebuggerEnvironment>PATH=$(ConanDebugPath);%PATH%</LocalDebuggerEnvironment>
            <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
            {% if ca_exclude %}
            <CAExcludePath>$(Conan{{name}}IncludeDirectories);$(CAExcludePath)</CAExcludePath>
            {% endif %}
          </PropertyGroup>
          <ItemDefinitionGroup>
            <ClCompile>
              <AdditionalIncludeDirectories>$(Conan{{name}}IncludeDirectories)%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
              <PreprocessorDefinitions>$(Conan{{name}}PreprocessorDefinitions)%(PreprocessorDefinitions)</PreprocessorDefinitions>
              <AdditionalOptions>$(Conan{{name}}CompilerFlags) %(AdditionalOptions)</AdditionalOptions>
            </ClCompile>
            <Link>
              <AdditionalLibraryDirectories>$(Conan{{name}}LibraryDirectories)%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
              <AdditionalDependencies>$(Conan{{name}}Libraries)%(AdditionalDependencies)</AdditionalDependencies>
              <AdditionalDependencies>$(Conan{{name}}SystemLibs)%(AdditionalDependencies)</AdditionalDependencies>
              <AdditionalOptions>$(Conan{{name}}LinkerFlags) %(AdditionalOptions)</AdditionalOptions>
            </Link>
            <Midl>
              <AdditionalIncludeDirectories>$(Conan{{name}}IncludeDirectories)%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
            </Midl>
            <ResourceCompile>
              <AdditionalIncludeDirectories>$(Conan{{name}}IncludeDirectories)%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
              <PreprocessorDefinitions>$(Conan{{name}}PreprocessorDefinitions)%(PreprocessorDefinitions)</PreprocessorDefinitions>
            </ResourceCompile>
          </ItemDefinitionGroup>
          {% else %}
          <PropertyGroup>
            <ExecutablePath>$(Conan{{name}}BinaryDirectories)$(ExecutablePath)</ExecutablePath>
          </PropertyGroup>
          {% endif %}
        </Project>
        """)

    def __init__(self, conanfile):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile
        #: Defines the build type. By default, the value of ``settings.build_type``.
        self.configuration = conanfile.settings.build_type
        #: Defines the configuration key used to conditionally select which property sheet to
        #: import (defaults to ``"Configuration"``).
        self.configuration_key = "Configuration"
        # TODO: This platform is not exactly the same as ``msbuild_arch``, because it differs
        # in x86=>Win32
        #: Platform name, e.g., ``Win32`` if ``settings.arch == "x86"``.
        self.platform = msvc_platform_from_arch(str(conanfile.settings.arch))
        #: Defines the platform key used to conditionally select which property sheet to
        #: import (defaults to ``"Platform"``).
        self.platform_key = "Platform"
        ca_exclude = "tools.microsoft.msbuilddeps:exclude_code_analysis"
        #: List of packages names patterns to add Visual Studio ``CAExcludePath`` property
        #: to each match as part of its ``conan_[DEP]_[CONFIG].props``. By default, value given by
        #: ``tools.microsoft.msbuilddeps:exclude_code_analysis`` configuration.
        self.exclude_code_analysis = self._conanfile.conf.get(ca_exclude, check_type=list)

    def generate(self):
        """
        Generates ``conan_<pkg>_<config>_vars.props``, ``conan_<pkg>_<config>.props``,
        and ``conan_<pkg>.props`` files into the ``conanfile.generators_folder``.
        """
        check_duplicated_generator(self, self._conanfile)
        if self.configuration is None:
            raise ConanException("MSBuildDeps.configuration is None, it should have a value")
        if self.platform is None:
            raise ConanException("MSBuildDeps.platform is None, it should have a value")
        generator_files = self._content()
        for generator_file, content in generator_files.items():
            save(generator_file, content)

    def _config_filename(self):
        props = [self.configuration,
                 self.platform]
        name = "".join("_%s" % v for v in props)
        return name.lower()

    def _condition(self):
        props = [(self.configuration_key, self.configuration),
                 (self.platform_key, self.platform)]
        condition = " And ".join("'$(%s)' == '%s'" % (k, v) for k, v in props)
        return condition

    @staticmethod
    def _dep_name(dep, build):
        dep_name = dep.ref.name
        if build:  # dep.context == CONTEXT_BUILD:
            dep_name += "_build"
        return MSBuildDeps._get_valid_xml_format(dep_name)

    @staticmethod
    def _get_valid_xml_format(name):
        return re.compile(r"[.+]").sub("_", name)

    def _vars_props_file(self, require, dep, name, cpp_info, build):
        """
        content for conan_vars_poco_x86_release.props, containing the variables for 1 config
        This will be for 1 package or for one component of a package
        :return: varfile content
        """

        def add_valid_ext(libname, libdirs=None):
            ext = os.path.splitext(libname)[1]
            if ext in VALID_LIB_EXTENSIONS:
                return f"{libname};"

            lib_name = f"{libname}.lib"
            if libdirs and not any(lib_name in os.listdir(d) for d in libdirs if os.path.isdir(d)):
                meson_name = f"lib{libname}.a"
                if any(meson_name in os.listdir(d) for d in libdirs if os.path.isdir(d)):
                    lib_name = meson_name
            return f"{lib_name};"

        pkg_placeholder = "$(Conan{}RootFolder)".format(name)

        def escape_path(path):
            # https://docs.microsoft.com/en-us/visualstudio/msbuild/
            #                          how-to-escape-special-characters-in-msbuild
            # https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-special-characters
            return path.lstrip("/")

        def join_paths(paths):
            # TODO: ALmost copied from CMakeDeps TargetDataContext
            ret = []
            for p in paths:
                assert os.path.isabs(p), "{} is not absolute".format(p)
                full_path = escape_path(p)
                if full_path.startswith(root_folder):
                    rel = full_path[len(root_folder)+1:]
                    full_path = ("%s/%s" % (pkg_placeholder, rel))
                ret.append(full_path)
            return "".join("{};".format(e) for e in ret)

        root_folder = dep.recipe_folder if dep.package_folder is None else dep.package_folder
        root_folder = escape_path(root_folder)
        # Make the root_folder relative to the generated conan_vars_xxx.props file
        relative_root_folder = relativize_path(root_folder, self._conanfile,
                                               "$(MSBuildThisFileDirectory)", normalize=False)

        bin_dirs = join_paths(cpp_info.bindirs)
        res_dirs = join_paths(cpp_info.resdirs)
        include_dirs = join_paths(cpp_info.includedirs)
        lib_dirs = join_paths(cpp_info.libdirs)
        libs = "".join([add_valid_ext(lib, cpp_info.libdirs) for lib in cpp_info.libs])
        # TODO: Missing objects
        system_libs = "".join([add_valid_ext(sys_dep) for sys_dep in cpp_info.system_libs])
        definitions = "".join("%s;" % d for d in cpp_info.defines)
        compiler_flags = " ".join(cpp_info.cxxflags + cpp_info.cflags)
        linker_flags = " ".join(cpp_info.sharedlinkflags + cpp_info.exelinkflags)

        # traits logic
        if require and not require.headers:
            include_dirs = ""
        if require and not require.libs:
            lib_dirs = ""
            libs = ""
        if require and not require.libs and not require.headers:
            definitions = ""
            compiler_flags = ""
            linker_flags = ""
        if require and not require.run:
            bin_dirs = ""

        fields = {
            'name': name,
            'root_folder': relative_root_folder,
            'bin_dirs': bin_dirs,
            'res_dirs': res_dirs,
            'include_dirs': include_dirs,
            'lib_dirs': lib_dirs,
            'libs': libs,
            # TODO: Missing objects
            'system_libs': system_libs,
            'definitions': definitions,
            'compiler_flags': compiler_flags,
            'linker_flags': linker_flags,
            'host_context': not build
        }
        formatted_template = Template(self._vars_props, trim_blocks=True,
                                      lstrip_blocks=True).render(**fields)
        return formatted_template

    def _activate_props_file(self, dep_name, vars_filename, deps, build):
        """
        Actual activation of the VS variables, per configuration
            - conan_pkgname_x86_release.props / conan_pkgname_compname_x86_release.props
        :param dep_name: pkgname / pkgname_compname
        :param deps: the name of other things to be included: [dep1, dep2:compA, ...]
        :param build: if it is a build require or not
        """

        # TODO: This must include somehow the user/channel, most likely pattern to exclude/include
        # Probably also the negation pattern, exclude all not @mycompany/*
        ca_exclude = any(fnmatch.fnmatch(dep_name, p) for p in self.exclude_code_analysis or ())
        template = Template(self._conf_props, trim_blocks=True, lstrip_blocks=True)
        content_multi = template.render(host_context=not build, name=dep_name, ca_exclude=ca_exclude,
                                        vars_filename=vars_filename, deps=deps)
        return content_multi

    @staticmethod
    def _dep_props_file(dep_name, filename, aggregated_filename, condition, content=None):
        """
        The file aggregating all configurations for a given pkg / component
            - conan_pkgname.props
        """
        # Current directory is the generators_folder
        if content:
            content_multi = content  # Useful for aggregating multiple components in one pass
        elif os.path.isfile(filename):
            content_multi = load(filename)
        else:
            content_multi = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
              <ImportGroup Label="PropertySheets">
              </ImportGroup>
              <PropertyGroup>
                <conan_{{name}}_props_imported>True</conan_{{name}}_props_imported>
              </PropertyGroup>
            </Project>
            """)
            content_multi = Template(content_multi).render({"name": dep_name})
        # parse the multi_file and add new import statement if needed
        dom = minidom.parseString(content_multi)
        import_vars = dom.getElementsByTagName('ImportGroup')[0]

        # Current vars
        children = import_vars.getElementsByTagName("Import")
        for node in children:
            if aggregated_filename == node.getAttribute("Project") \
                    and condition == node.getAttribute("Condition"):
                break
        else:  # create a new import statement
            import_node = dom.createElement('Import')
            import_node.setAttribute('Condition', condition)
            import_node.setAttribute('Project', aggregated_filename)
            import_vars.appendChild(import_node)

        content_multi = dom.toprettyxml()
        content_multi = "\n".join(line for line in content_multi.splitlines() if line.strip())
        return content_multi

    def _conandeps(self):
        """ this is a .props file including direct declared dependencies
        """
        # Current directory is the generators_folder
        conandeps_filename = "conandeps.props"
        direct_deps = self._conanfile.dependencies.filter({"direct": True})
        pkg_aggregated_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
              <ImportGroup Label="PropertySheets">
              </ImportGroup>
            </Project>
            """)
        for req, dep in direct_deps.items():
            dep_name = self._dep_name(dep, req.build)
            filename = "conan_%s.props" % dep_name
            comp_condition = "'$(conan_%s_props_imported)' != 'True'" % dep_name
            pkg_aggregated_content = self._dep_props_file("", conandeps_filename, filename,
                                                          condition=comp_condition,
                                                          content=pkg_aggregated_content)
        return {conandeps_filename: pkg_aggregated_content}

    def _package_props_files(self, require, dep, build=False):
        """ all the files for a given package:
        - conan_pkgname_vars_config.props: definition of variables, one per config
        - conan_pkgname_config.props: The one using those variables. This is very different for
                                      Host and build, build only activate <ExecutablePath>
        - conan_pkgname.props: Conditional aggregate xxx_config.props based on active config
        """
        conf_name = self._config_filename()
        condition = self._condition()
        dep_name = self._dep_name(dep, build)
        result = {}
        pkg_deps = get_transitive_requires(self._conanfile, dep)  # only non-skipped dependencies
        if dep.cpp_info.has_components:
            pkg_aggregated_content = None
            for comp_name, comp_info in dep.cpp_info.components.items():
                full_comp_name = "{}_{}".format(dep_name, self._get_valid_xml_format(comp_name))
                vars_filename = "conan_%s_vars%s.props" % (full_comp_name, conf_name)
                activate_filename = "conan_%s%s.props" % (full_comp_name, conf_name)
                comp_filename = "conan_%s.props" % full_comp_name
                pkg_filename = "conan_%s.props" % dep_name

                public_deps = []  # To store the xml dependencies/file names
                for required_pkg, required_comp in comp_info.parsed_requires():
                    if required_pkg is not None:  # Points to a component of a different package
                        try:
                            required = pkg_deps[required_pkg]
                        except KeyError:  # The transitive dep might have been skipped
                            required = None
                        if required:  # The transitive dep might have been skipped
                            required_name = required.ref.name
                            public_deps.append(required_name if required_pkg == required_comp
                                               else "{}_{}".format(required_name, required_comp))
                    else:  # Points to a component of same package
                        public_deps.append("{}_{}".format(dep_name, required_comp))
                public_deps = [self._get_valid_xml_format(d) for d in public_deps]
                result[vars_filename] = self._vars_props_file(require, dep, full_comp_name,
                                                              comp_info, build=build)
                result[activate_filename] = self._activate_props_file(full_comp_name, vars_filename,
                                                                      public_deps, build=build)
                result[comp_filename] = self._dep_props_file(full_comp_name, comp_filename,
                                                             activate_filename, condition)
                comp_condition = "'$(conan_%s_props_imported)' != 'True'" % full_comp_name
                pkg_aggregated_content = self._dep_props_file(dep_name, pkg_filename, comp_filename,
                                                              condition=comp_condition,
                                                              content=pkg_aggregated_content)
                result[pkg_filename] = pkg_aggregated_content
        else:
            cpp_info = dep.cpp_info
            vars_filename = "conan_%s_vars%s.props" % (dep_name, conf_name)
            activate_filename = "conan_%s%s.props" % (dep_name, conf_name)
            pkg_filename = "conan_%s.props" % dep_name
            public_deps = [self._dep_name(d, build) for d in pkg_deps.values()]

            result[vars_filename] = self._vars_props_file(require, dep, dep_name, cpp_info,
                                                          build=build)
            result[activate_filename] = self._activate_props_file(dep_name, vars_filename,
                                                                  public_deps, build=build)
            result[pkg_filename] = self._dep_props_file(dep_name, pkg_filename, activate_filename,
                                                        condition=condition)
        return result

    def _content(self):
        if not self._conanfile.settings.get_safe("build_type"):
            raise ConanException("The 'msbuild' generator requires a 'build_type' setting value")
        result = {}

        for req, dep in self._conanfile.dependencies.host.items():
            result.update(self._package_props_files(req, dep, build=False))
        for req, dep in self._conanfile.dependencies.test.items():
            result.update(self._package_props_files(req, dep, build=False))
        for req, dep in self._conanfile.dependencies.build.items():
            result.update(self._package_props_files(req, dep, build=True))

        # Include all direct build_requires for host context. This might change
        result.update(self._conandeps())

        return result
