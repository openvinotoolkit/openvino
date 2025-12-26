import textwrap

from conan.tools.cmake.cmakedeps.templates import CMakeDepsFileTemplate

"""

FooConfig.cmake
foo-config.cmake

"""


class ConfigTemplate(CMakeDepsFileTemplate):

    @property
    def filename(self):
        if self.generating_module:
            return "Find{}.cmake".format(self.file_name)
        else:
            if self.file_name == self.file_name.lower():
                return "{}-config.cmake".format(self.file_name)
            else:
                return "{}Config.cmake".format(self.file_name)

    @property
    def additional_variables_prefixes(self):
        prefix_list = (
            self.cmakedeps.get_property("cmake_additional_variables_prefixes", self.conanfile, check_type=list) or [])
        return list(set([self.file_name] + prefix_list))

    @property
    def parsed_extra_variables(self):
        # Reading configuration from "cmake_extra_variables" property
        from conan.tools.cmake.utils import parse_extra_variable
        conf_extra_variables = self.conanfile.conf.get("tools.cmake.cmaketoolchain:extra_variables",
                                                       default={}, check_type=dict)
        dep_extra_variables = self.cmakedeps.get_property("cmake_extra_variables", self.conanfile,
                                                          check_type=dict) or {}
        # The configuration variables have precedence over the dependency ones
        extra_variables = {dep: value for dep, value in dep_extra_variables.items()
                           if dep not in conf_extra_variables}
        parsed_extra_variables = {}
        for key, value in extra_variables.items():
            parsed_extra_variables[key] = parse_extra_variable("cmake_extra_variables",
                                                               key, value)
        return parsed_extra_variables


    @property
    def context(self):
        targets_include = "" if not self.generating_module else "module-"
        targets_include += "{}Targets.cmake".format(self.file_name)
        return {"is_module": self.generating_module,
                "version": self.conanfile.ref.version,
                "file_name":  self.file_name,
                "additional_variables_prefixes": self.additional_variables_prefixes,
                "extra_variables": self.parsed_extra_variables,
                "pkg_name": self.pkg_name,
                "config_suffix": self.config_suffix,
                "check_components_exist": self.cmakedeps.check_components_exist,
                "targets_include_file": targets_include}

    @property
    def template(self):
        return textwrap.dedent("""\
        {%- macro pkg_var(pkg_name, var, config_suffix) -%}
             {{'${'+pkg_name+'_'+var+config_suffix+'}'}}
        {%- endmacro -%}
        ########## MACROS ###########################################################################
        #############################################################################################

        # Requires CMake > 3.15
        if(${CMAKE_VERSION} VERSION_LESS "3.15")
            message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
        endif()

        if({{ file_name }}_FIND_QUIETLY)
            set({{ file_name }}_MESSAGE_MODE VERBOSE)
        else()
            set({{ file_name }}_MESSAGE_MODE STATUS)
        endif()

        include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
        include(${CMAKE_CURRENT_LIST_DIR}/{{ targets_include_file }})
        include(CMakeFindDependencyMacro)

        check_build_type_defined()

        foreach(_DEPENDENCY {{ pkg_var(pkg_name, 'FIND_DEPENDENCY_NAMES', '') }} )
            # Check that we have not already called a find_package with the transitive dependency
            if(NOT {{ '${_DEPENDENCY}' }}_FOUND)
                find_dependency({{ '${_DEPENDENCY}' }} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
            endif()
        endforeach()

        {% for prefix in additional_variables_prefixes %}
        set({{ prefix }}_VERSION_STRING "{{ version }}")
        set({{ prefix }}_INCLUDE_DIRS {{ pkg_var(pkg_name, 'INCLUDE_DIRS', config_suffix) }} )
        set({{ prefix }}_INCLUDE_DIR {{ pkg_var(pkg_name, 'INCLUDE_DIRS', config_suffix) }} )
        set({{ prefix }}_LIBRARIES {{ pkg_var(pkg_name, 'LIBRARIES', config_suffix) }} )
        set({{ prefix }}_DEFINITIONS {{ pkg_var(pkg_name, 'DEFINITIONS', config_suffix) }} )

        {% endfor %}

        # Definition of extra CMake variables from cmake_extra_variables

        {% for key, value in extra_variables.items() %}
        set({{ key }} {{ value }})
        {% endfor %}

        # Only the last installed configuration BUILD_MODULES are included to avoid the collision
        foreach(_BUILD_MODULE {{ pkg_var(pkg_name, 'BUILD_MODULES_PATHS', config_suffix) }} )
            message({% raw %}${{% endraw %}{{ file_name }}_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
            include({{ '${_BUILD_MODULE}' }})
        endforeach()

        {% if check_components_exist %}
        # Check that the specified components in the find_package(Foo COMPONENTS x y z) are there
        # This is the variable filled by CMake with the requested components in find_package
        if({{ file_name }}_FIND_COMPONENTS)
            foreach(_FIND_COMPONENT {{ pkg_var(file_name, 'FIND_COMPONENTS', '') }})
                if (TARGET ${_FIND_COMPONENT})
                    message({% raw %}${{% endraw %}{{ file_name }}_MESSAGE_MODE} "Conan: Component '${_FIND_COMPONENT}' found in package '{{ pkg_name }}'")
                else()
                    message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package '{{ pkg_name }}'")
                endif()
            endforeach()
        endif()
        {% endif %}

        {% if is_module %}
        include(FindPackageHandleStandardArgs)
        set({{ file_name }}_FOUND 1)
        set({{ file_name }}_VERSION "{{ version }}")

        find_package_handle_standard_args({{ file_name }}
                                          REQUIRED_VARS {{ file_name }}_VERSION
                                          VERSION_VAR {{ file_name }}_VERSION)
        mark_as_advanced({{ file_name }}_FOUND {{ file_name }}_VERSION)

        {% for prefix in additional_variables_prefixes %}
        set({{ prefix }}_FOUND 1)
        set({{ prefix }}_VERSION "{{ version }}")
        mark_as_advanced({{ prefix }}_FOUND {{ prefix }}_VERSION)
        {% endfor %}

        {% endif %}
        """)
