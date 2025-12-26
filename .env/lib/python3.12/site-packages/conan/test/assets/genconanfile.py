from conan.api.model import RecipeReference


class GenConanfile(object):
    """
    USAGE:

    x = GenConanfile().with_import("import os").\
        with_setting("os").\
        with_option("shared", [True, False]).\
        with_default_option("shared", True).\
        with_build_msg("holaaa").\
        with_build_msg("adiooos").\
        with_package_file("file.txt", "hola").\
        with_package_file("file2.txt", "hola")
    """

    def __init__(self, name=None, version=None):
        self._imports = ["from conan import ConanFile"]
        self._name = name
        self._version = version
        self._package_type = None
        self._settings = None
        self._options = None
        self._generators = None
        self._default_options = None
        self._provides = None
        self._deprecated = None
        self._package_lines = None
        self._finalize_lines = None
        self._package_files = None
        self._package_files_env = None
        self._package_files_link = None
        self._build_messages = None
        self._requires = None
        self._requirements = None
        self._python_requires = None
        self._build_requires = None
        self._build_requirements = None
        self._tool_requires = None
        self._tool_requirements = None
        self._test_requires = None
        self._revision_mode = None
        self._package_info = None
        self._package_id_lines = None
        self._test_lines = None
        self._exports_sources = None
        self._exports = None
        self._cmake_build = False
        self._class_attributes = None

        self._is_tested_ref_build_require = None

    def with_package_type(self, value):
        self._package_type = value
        return self

    def with_name(self, name):
        self._name = name
        return self

    def with_version(self, version):
        self._version = version
        return self

    def with_provides(self, provides):
        self._provides = self._provides or []
        self._provides.append(provides)
        return self

    def with_deprecated(self, deprecated):
        self._deprecated = deprecated
        return self

    def with_revision_mode(self, revision_mode):
        self._revision_mode = revision_mode
        return self

    def with_generator(self, generator):
        self._generators = self._generators or []
        self._generators.append(generator)
        return self

    def with_exports_sources(self, *exports):
        self._exports_sources = self._exports_sources or []
        for export in exports:
            self._exports_sources.append(export)
        return self

    def with_exports(self, *exports):
        self._exports = self._exports or []
        for export in exports:
            self._exports.append(export)
        return self

    def with_require(self, ref):
        self._requires = self._requires or []
        ref_str = self._get_full_ref_str(ref)
        self._requires.append(ref_str)
        return self

    def with_requires(self, *refs):
        for ref in refs:
            self.with_require(ref)
        return self

    @staticmethod
    def _get_full_ref_str(ref):
        if isinstance(ref, RecipeReference):
            ref_str = ref.repr_notime()
        else:
            ref_str = ref
        return ref_str

    def with_requirement(self, ref, **kwargs):
        self._requirements = self._requirements or []
        ref_str = self._get_full_ref_str(ref)
        self._requirements.append((ref_str, kwargs))
        return self

    def with_build_requires(self, *refs):
        self._build_requires = self._build_requires or []
        for ref in refs:
            ref_str = self._get_full_ref_str(ref)
            self._build_requires.append(ref_str)
        return self

    def with_python_requires(self, *refs):
        self._python_requires = self._python_requires or []
        for ref in refs:
            ref_str = self._get_full_ref_str(ref)
            self._python_requires.append(ref_str)
        return self

    def with_tool_requires(self, *refs):
        self._tool_requires = self._tool_requires or []
        for ref in refs:
            ref_str = self._get_full_ref_str(ref)
            self._tool_requires.append(ref_str)
        return self

    def with_test_requires(self, *refs):
        self._test_requires = self._test_requires or []
        for ref in refs:
            ref_str = self._get_full_ref_str(ref)
            self._test_requires.append(ref_str)
        return self

    def with_test_reference_as_build_require(self):
        self._is_tested_ref_build_require = True
        return self

    def with_build_requirement(self, ref, **kwargs):
        self._build_requirements = self._build_requirements or []
        ref_str = self._get_full_ref_str(ref)
        self._build_requirements.append((ref_str, kwargs))
        return self

    def with_tool_requirement(self, ref, **kwargs):
        self._tool_requirements = self._tool_requirements or []
        ref_str = self._get_full_ref_str(ref)
        self._tool_requirements.append((ref_str, kwargs))
        return self

    def with_import(self, *imports):
        for i in imports:
            if i not in self._imports:
                self._imports.append(i)
        return self

    def with_setting(self, setting):
        self._settings = self._settings or []
        self._settings.append(setting)
        return self

    def with_settings(self, *settings):
        self._settings = self._settings or []
        self._settings.extend(settings)
        return self

    def with_option(self, option_name, values, default=None):
        self._options = self._options or {}
        self._options[option_name] = values
        if default is not None:
            self.with_default_option(option_name, default)
        return self

    def with_default_option(self, option_name, value):
        self._default_options = self._default_options or {}
        self._default_options[option_name] = value
        return self

    def with_shared_option(self, default=False):
        return self.with_option("shared", [True, False]).with_default_option("shared", default)

    def with_package_file(self, file_name, contents=None, env_var=None, link=None):
        if not contents and not env_var:
            raise Exception("Specify contents or env_var")
        self._package_files = self._package_files or {}
        self._package_files_link = self._package_files_link or {}
        self._package_files_env = self._package_files_env or {}
        self.with_import("import os")
        self.with_import("from conan.tools.files import save, chdir")
        if contents:
            self._package_files[file_name] = contents
        if link:
            self._package_files_link[file_name] = link
        if env_var:
            self._package_files_env[file_name] = env_var
        return self

    def with_package(self, *lines):
        self._package_lines = self._package_lines or []
        for line in lines:
            self._package_lines.append(line)
        return self

    def with_finalize(self, *lines):
        self._finalize_lines = self._finalize_lines or []
        for line in lines:
            self._finalize_lines.append(line)
        return self

    def with_build_msg(self, msg):
        self._build_messages = self._build_messages or []
        self._build_messages.append(msg)
        return self

    def with_package_info(self, cpp_info=None, env_info=None):
        assert cpp_info is None or isinstance(cpp_info, dict), "cpp_info ({}) expects dict".format(type(cpp_info))
        assert env_info is None or isinstance(env_info, dict), "env_info ({}) expects dict".format(type(env_info))
        self._package_info = self._package_info or {}
        if cpp_info:
            self._package_info["cpp_info"] = cpp_info
        if env_info:
            self._package_info["env_info"] = env_info
        return self

    def with_package_id(self, *lines):
        self._package_id_lines = self._package_id_lines or []
        for line in lines:
            self._package_id_lines.append(line)
        return self

    def with_test(self, *lines):
        self._test_lines = self._test_lines or []
        for line in lines:
            self._test_lines.append(line)
        return self

    def with_cmake_build(self):
        self._imports.append("from conan.tools.cmake import CMake")
        self._generators = self._generators or []
        self._generators.append("CMakeDeps")
        self._generators.append("CMakeToolchain")
        self.with_settings("os", "compiler", "arch", "build_type")
        self._cmake_build = True
        return self

    def with_class_attribute(self, attr):
        """.with_class_attribute("no_copy_sources=True") """
        self._class_attributes = self._class_attributes or []
        self._class_attributes.append(attr)
        return self

    @property
    def _name_render(self):
        return "name = '{}'".format(self._name)

    @property
    def _version_render(self):
        return "version = '{}'".format(self._version)

    @property
    def _package_type_render(self):
        return "package_type = '{}'".format(self._package_type)

    @property
    def _provides_render(self):
        line = ", ".join('"{}"'.format(provide) for provide in self._provides)
        return "provides = {}".format(line)

    @property
    def _deprecated_render(self):
        return "deprecated = {}".format(self._deprecated)

    @property
    def _generators_render(self):
        line = ", ".join('"{}"'.format(generator) for generator in self._generators)
        return "generators = {}".format(line)

    @property
    def _revision_mode_render(self):
        line = "revision_mode=\"{}\"".format(self._revision_mode)
        return line

    @property
    def _settings_render(self):
        line = ", ".join('"%s"' % s for s in self._settings)
        return "settings = {}".format(line)

    @property
    def _options_render(self):
        line = ", ".join('"%s": %s' % (k, v) for k, v in self._options.items())
        tmp = "options = {%s}" % line
        return tmp

    @property
    def _default_options_render(self):
        line = ", ".join('"%s": %s' % (k, v) for k, v in self._default_options.items())
        tmp = "default_options = {%s}" % line
        return tmp

    @property
    def _build_requirements_render(self):
        lines = []
        for ref, kwargs in self._build_requirements:
            args = ", ".join("{}={}".format(k, f'"{v}"' if not isinstance(v, (bool, dict)) else v)
                             for k, v in kwargs.items())
            lines.append('        self.build_requires("{}", {})'.format(ref, args))
        return "def build_requirements(self):\n{}\n".format("\n".join(lines))

    @property
    def _build_requires_render(self):
        line = ", ".join(['"{}"'.format(r) for r in self._build_requires])
        tmp = "build_requires = %s" % line
        return tmp

    @property
    def _python_requires_render(self):
        line = ", ".join(['"{}"'.format(r) for r in self._python_requires])
        tmp = "python_requires = %s" % line
        return tmp

    @property
    def _tool_requires_render(self):
        line = ", ".join(['"{}"'.format(r) for r in self._tool_requires])
        tmp = "tool_requires = %s" % line
        return tmp

    @property
    def _requires_render(self):
        items = []
        for ref in self._requires:
            items.append('"{}"'.format(ref))
        return "requires = ({}, )".format(", ".join(items))

    @property
    def _test_requires_render(self):
        line = ", ".join(['"{}"'.format(r) for r in self._test_requires])
        tmp = "test_requires = {}".format(line)
        return tmp

    @property
    def _requirements_render(self):
        lines = ["", "    def requirements(self):"]
        for ref, kwargs in self._requirements or []:
            args = ", ".join("{}={}".format(k, f'"{v}"' if isinstance(v, str) else v)
                             for k, v in kwargs.items())
            lines.append('        self.requires("{}", {})'.format(ref, args))

        for ref, kwargs in self._build_requirements or []:
            args = ", ".join("{}={}".format(k, f'"{v}"' if not isinstance(v, (bool, dict)) else v)
                             for k, v in kwargs.items())
            lines.append('        self.build_requires("{}", {})'.format(ref, args))

        for ref, kwargs in self._tool_requirements or []:
            args = ", ".join("{}={}".format(k, f'"{v}"' if not isinstance(v, (bool, dict)) else v)
                             for k, v in kwargs.items())
            lines.append('        self.tool_requires("{}", {})'.format(ref, args))

        return "\n".join(lines)

    @property
    def _package_method(self):
        return (self._package_lines or self._package_files or self._package_files_env or
                self._package_files_link)

    @property
    def _finalize_method(self):
        return self._finalize_lines

    @property
    def _package_method_render(self):
        lines = []
        if self._package_lines:
            lines.extend("        {}".format(line) for line in self._package_lines)
        if self._package_files:
            lines = ['        save(self, os.path.join(self.package_folder, "{}"), "{}")'
                     ''.format(key, value)
                     for key, value in self._package_files.items()]

        if self._package_files_env:
            lines.extend(['        save(self, os.path.join(self.package_folder, "{}"), '
                          'os.getenv("{}"))'.format(key, value)
                          for key, value in self._package_files_env.items()])
        if self._package_files_link:
            lines.extend(['        with chdir(self, os.path.dirname('
                          'os.path.join(self.package_folder, "{}"))):\n'
                          '            os.symlink(os.path.basename("{}"), '
                          'os.path.join(self.package_folder, "{}"))'.format(key, key, value)
                          for key, value in self._package_files_link.items()])

        if not lines:
            return ""
        return """
    def package(self):
{}
    """.format("\n".join(lines))

    @property
    def _finalize_method_render(self):
        lines = []
        if self._finalize_lines:
            lines.extend("        {}".format(line) for line in self._finalize_lines)

        if not lines:
            return ""
        return """
    def finalize(self):
{}
        """.format("\n".join(lines))

    @property
    def _build_render(self):
        if not self._build_messages and not self._cmake_build:
            return None
        lines = []
        if self._build_messages:
            lines = ['        self.output.warning("{}")'.format(m) for m in self._build_messages]
        if self._cmake_build:
            lines.extend(['        cmake = CMake(self)',
                          '        cmake.configure()',
                          '        cmake.build()'])
        return """
    def build(self):
{}
    """.format("\n".join(lines))

    @property
    def _package_info_render(self):
        lines = []
        if "cpp_info" in self._package_info:
            for k, v in self._package_info["cpp_info"].items():
                if k == "components":
                    for comp_name, comp in v.items():
                        for comp_attr_name, comp_attr_value in comp.items():
                            lines.append('        self.cpp_info.components["{}"].{} = {}'.format(
                                comp_name, comp_attr_name, str(comp_attr_value)))
                else:
                    lines.append('        self.cpp_info.{} = {}'.format(k, str(v)))

        return """
    def package_info(self):
{}
        """.format("\n".join(lines))

    @property
    def _package_id_lines_render(self):
        lines = ['        {}'.format(line) for line in self._package_id_lines]
        return """
    def package_id(self):
{}
        """.format("\n".join(lines))

    @property
    def _test_lines_render(self):
        if self._is_tested_ref_build_require:
            lines = ["",
                     "    def build_requirements(self):",
                     '         self.tool_requires(self.tested_reference_str)',
                     "",
                     '    def test(self):']
        else:
            lines = ["",
                     "    def requirements(self):",
                     '         self.requires(self.tested_reference_str)',
                     "",
                     '    def test(self):']

        lines += ['        %s' % m for m in self._test_lines]
        return "\n".join(lines)

    @property
    def _exports_sources_render(self):
        line = ", ".join('"{}"'.format(e) for e in self._exports_sources)
        return "exports_sources = {}".format(line)

    @property
    def _exports_render(self):
        line = ", ".join('"{}"'.format(e) for e in self._exports)
        return "exports = {}".format(line)

    @property
    def _class_attributes_render(self):
        self._class_attributes = self._class_attributes or []
        return ["    {}".format(a) for a in self._class_attributes]

    def __repr__(self):
        ret = []
        ret.extend(self._imports)
        ret.append("class HelloConan(ConanFile):")

        for member in ("name", "version", "package_type", "provides", "deprecated",
                       "exports_sources", "exports", "generators", "requires", "build_requires",
                       "tool_requires", "test_requires", "requirements", "python_requires",
                       "revision_mode", "settings", "options", "default_options", "build",
                       "package_method", "package_info", "package_id_lines", "test_lines",
                       "finalize_method"
                       ):
            if member == "requirements":
                # FIXME: This seems exclusive, but we could mix them?
                v = self._requirements or self._tool_requirements or self._build_requirements
            else:
                v = getattr(self, "_{}".format(member), None)
            if v is not None:
                ret.append("    {}".format(getattr(self, "_{}_render".format(member))))

        ret.extend(self._class_attributes_render)
        build = self._build_render
        if build is not None:
            ret.append("    {}".format(self._build_render))
        if ret[-1] == "class HelloConan(ConanFile):":
            ret.append("    pass")
        return "\n".join(ret)
