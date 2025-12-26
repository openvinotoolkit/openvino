import textwrap

import jinja2
from jinja2 import Template

from conan.api.output import Color
from conan.internal import check_duplicated_generator
from conan.tools.cmake.cmakedeps import FIND_MODE_CONFIG, FIND_MODE_NONE, FIND_MODE_BOTH, \
    FIND_MODE_MODULE
from conan.tools.cmake.cmakedeps.templates.config import ConfigTemplate
from conan.tools.cmake.cmakedeps.templates.config_version import ConfigVersionTemplate
from conan.tools.cmake.cmakedeps.templates.macros import MacrosTemplate
from conan.tools.cmake.cmakedeps.templates.target_configuration import TargetConfigurationTemplate
from conan.tools.cmake.cmakedeps.templates.target_data import ConfigDataTemplate
from conan.tools.cmake.cmakedeps.templates.targets import TargetsTemplate
from conan.tools.files import save
from conan.errors import ConanException
from conan.internal.model.dependencies import get_transitive_requires


class CMakeDeps(object):

    def __init__(self, conanfile):
        self._conanfile = conanfile
        self.arch = self._conanfile.settings.get_safe("arch")
        self.configuration = str(self._conanfile.settings.build_type)

        # Activate the build config files for the specified libraries
        self.build_context_activated = []
        # By default, the build modules are generated for host context only
        self.build_context_build_modules = []
        # If specified, the files/targets/variables for the build context will be renamed appending
        # a suffix. It is necessary in case of same require and build_require and will cause an error
        self.build_context_suffix = {}

        # Enable/Disable checking if a component target exists or not
        self.check_components_exist = False
        self._properties = {}

    def generate(self):
        """
        This method will save the generated files to the ``conanfile.generators_folder`` folder
        """
        check_duplicated_generator(self, self._conanfile)

        # Current directory is the generators_folder
        generator_files = self.content
        for generator_file, content in generator_files.items():
            save(self._conanfile, generator_file, content)
        self.generate_aggregator()

    @property
    def content(self):
        macros = MacrosTemplate()
        ret = {macros.filename: macros.render()}

        host_req = self._conanfile.dependencies.host
        build_req = self._conanfile.dependencies.direct_build
        test_req = self._conanfile.dependencies.test

        # Check if the same package is at host and build and the same time
        activated_br = {r.ref.name for r in build_req.values()
                        if r.ref.name in self.build_context_activated}
        common_names = {r.ref.name for r in host_req.values()}.intersection(activated_br)
        for common_name in common_names:
            suffix = self.build_context_suffix.get(common_name)
            if not suffix:
                raise ConanException("The package '{}' exists both as 'require' and as "
                                     "'build require'. You need to specify a suffix using the "
                                     "'build_context_suffix' attribute at the CMakeDeps "
                                     "generator.".format(common_name))

        # Iterate all the transitive requires
        direct_configs = []
        for require, dep in list(host_req.items()) + list(build_req.items()) + list(test_req.items()):
            # Require is not used at the moment, but its information could be used,
            # and will be used in Conan 2.0
            # Filter the build_requires not activated with cmakedeps.build_context_activated
            if require.build and dep.ref.name not in self.build_context_activated:
                continue

            cmake_find_mode = self.get_property("cmake_find_mode", dep)
            cmake_find_mode = cmake_find_mode or FIND_MODE_CONFIG
            cmake_find_mode = cmake_find_mode.lower()
            # Skip from the requirement
            if cmake_find_mode == FIND_MODE_NONE:
                # Skip the generation of config files for this node, it will be located externally
                continue

            if cmake_find_mode in (FIND_MODE_CONFIG, FIND_MODE_BOTH):
                self._generate_files(require, dep, ret, find_module_mode=False)

            if cmake_find_mode in (FIND_MODE_MODULE, FIND_MODE_BOTH):
                self._generate_files(require, dep, ret, find_module_mode=True)

            if require.direct:  # aggregate config information for user convenience
                find_module_mode = True if cmake_find_mode == FIND_MODE_MODULE else False
                config = ConfigTemplate(self, require, dep, find_module_mode)
                direct_configs.append(config)

        if direct_configs:
            # Some helpful verbose messages about generated files
            msg = ["CMakeDeps necessary find_package() and targets for your CMakeLists.txt"]
            for config in direct_configs:
                msg.append(f"    find_package({config.file_name})")
            targets = ' '.join(c.root_target_name for c in direct_configs)
            msg.append(f"    target_link_libraries(... {targets})")
            if self._conanfile._conan_is_consumer:  # noqa
                self._conanfile.output.info("\n".join(msg), fg=Color.CYAN)
            else:
                self._conanfile.output.verbose("\n".join(msg))

        return ret

    def _generate_files(self, require, dep, ret, find_module_mode):
        if not find_module_mode:
            config_version = ConfigVersionTemplate(self, require, dep)
            ret[config_version.filename] = config_version.render()

        data_target = ConfigDataTemplate(self, require, dep, find_module_mode)
        data_content = data_target.render()
        ret[data_target.filename] = data_content

        target_configuration = TargetConfigurationTemplate(self, require, dep, find_module_mode)
        ret[target_configuration.filename] = target_configuration.render()

        targets = TargetsTemplate(self, require, dep, find_module_mode)
        ret[targets.filename] = targets.render()

        config = ConfigTemplate(self, require, dep, find_module_mode)
        # Only the latest configuration BUILD_MODULES and additional_variables will be used
        # in multi-config they will be overwritten by the latest install
        ret[config.filename] = config.render()

    def set_property(self, dep, prop, value, build_context=False):
        """
        Using this method you can overwrite the :ref:`property<CMakeDeps Properties>` values set by
        the Conan recipes from the consumer. This can be done for ``cmake_file_name``, ``cmake_target_name``,
        ``cmake_find_mode``, ``cmake_module_file_name``, ``cmake_module_target_name``, ``cmake_additional_variables_prefixes``,
        ``cmake_config_version_compat``, ``system_package_version``,
        ``cmake_build_modules``, ``nosoname``, ``cmake_target_aliases`` and ``cmake_extra_variables``.

        :param dep: Name of the dependency to set the :ref:`property<CMakeDeps Properties>`. For
         components use the syntax: ``dep_name::component_name``.
        :param prop: Name of the :ref:`property<CMakeDeps Properties>`.
        :param value: Value of the property. Use ``None`` to invalidate any value set by the
         upstream recipe.
        :param build_context: Set to ``True`` if you want to set the property for a dependency that
         belongs to the build context (``False`` by default).
        """
        build_suffix = "&build" if build_context else ""
        self._properties.setdefault(f"{dep}{build_suffix}", {}).update({prop: value})

    def get_property(self, prop, dep, comp_name=None, check_type=None):
        dep_name = dep.ref.name
        build_suffix = "&build" if str(
            dep_name) in self.build_context_activated and dep.context == "build" else ""
        dep_comp = f"{str(dep_name)}::{comp_name}" if comp_name else f"{str(dep_name)}"
        try:
            value = self._properties[f"{dep_comp}{build_suffix}"][prop]
            if check_type is not None and not isinstance(value, check_type):
                raise ConanException(
                    f'The expected type for {prop} is "{check_type.__name__}", but "{type(value).__name__}" was found')
            return value
        except KeyError:
            return dep.cpp_info.get_property(prop, check_type=check_type) if not comp_name \
                else dep.cpp_info.components[comp_name].get_property(prop, check_type=check_type)

    def get_cmake_package_name(self, dep, module_mode=None):
        """Get the name of the file for the ``find_package(XXX)`` call"""
        # This is used by CMakeDeps to determine:
        # - The filename to generate (XXX-config.cmake or FindXXX.cmake)
        # - The name of the defined XXX_DIR variables
        # - The name of transitive dependencies for calls to find_dependency
        if module_mode and self.get_find_mode(dep) in [FIND_MODE_MODULE, FIND_MODE_BOTH]:
            ret = self.get_property("cmake_module_file_name", dep)
            if ret:
                return ret
        ret = self.get_property("cmake_file_name", dep)
        return ret or dep.ref.name

    def get_find_mode(self, dep):
        """
        :param dep: requirement
        :return: One of ``"none"``, ``"config"``, ``"module"`` or ``"both"``. Defaults to ``"config"`` when not set
        """
        tmp = self.get_property("cmake_find_mode", dep)
        if tmp is None:
            return "config"
        return tmp.lower()

    def generate_aggregator(self):
        host = self._conanfile.dependencies.host
        build_req = self._conanfile.dependencies.direct_build
        test_req = self._conanfile.dependencies.test

        configs = []
        for require, dep in list(host.items()) + list(build_req.items()) + list(test_req.items()):
            if not require.direct:
                continue
            if require.build and dep.ref.name not in self.build_context_activated:
                continue
            cmake_find_mode = self.get_property("cmake_find_mode", dep)
            cmake_find_mode = cmake_find_mode or FIND_MODE_CONFIG
            cmake_find_mode = cmake_find_mode.lower()
            find_module_mode = True if cmake_find_mode == FIND_MODE_MODULE else False
            config = ConfigTemplate(self, require, dep, find_module_mode)
            configs.append(config)

        template = textwrap.dedent("""\
            message(STATUS "Conan: Using CMakeDeps conandeps_legacy.cmake aggregator via include()")
            message(STATUS "Conan: It is recommended to use explicit find_package() per dependency instead")

            {% for config in configs %}
            find_package({{config.file_name}})
            {% endfor %}

            set(CONANDEPS_LEGACY {% for t in configs %} {{t.root_target_name}} {% endfor %})
            """)

        template = Template(template, trim_blocks=True, lstrip_blocks=True,
                            undefined=jinja2.StrictUndefined)
        conandeps = template.render({"configs": configs})
        save(self._conanfile, "conandeps_legacy.cmake", conandeps)

    def get_transitive_requires(self, conanfile):
        # Prepared to filter transitive tool-requires with visible=True
        return get_transitive_requires(self._conanfile, conanfile)
