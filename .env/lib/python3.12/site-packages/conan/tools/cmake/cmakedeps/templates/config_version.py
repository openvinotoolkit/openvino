import textwrap

from conan.tools.cmake.cmakedeps.templates import CMakeDepsFileTemplate
from conan.errors import ConanException

"""
    foo-config-version.cmake

"""


class ConfigVersionTemplate(CMakeDepsFileTemplate):

    @property
    def filename(self):
        if self.file_name == self.file_name.lower():
            return "{}-config-version.cmake".format(self.file_name)
        else:
            return "{}ConfigVersion.cmake".format(self.file_name)

    @property
    def context(self):
        policy = self.cmakedeps.get_property("cmake_config_version_compat", self.conanfile)
        if policy is None:
            policy = "SameMajorVersion"
        if policy not in ("AnyNewerVersion", "SameMajorVersion", "SameMinorVersion", "ExactVersion"):
            raise ConanException(f"Unknown cmake_config_version_compat={policy} in {self.conanfile}")
        version = self.cmakedeps.get_property("system_package_version", self.conanfile)
        version = version or self.conanfile.ref.version
        return {"version": version,
                "policy": policy}

    @property
    def template(self):
        # https://gitlab.kitware.com/cmake/cmake/blob/master/Modules/BasicConfigVersion-SameMajorVersion.cmake.in
        # This will be at XXX-config-version.cmake
        # AnyNewerVersion|SameMajorVersion|SameMinorVersion|ExactVersion
        ret = textwrap.dedent("""\
            set(PACKAGE_VERSION "{{ version }}")

            if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
                set(PACKAGE_VERSION_COMPATIBLE FALSE)
            else()
                {% if policy == "AnyNewerVersion" %}
                set(PACKAGE_VERSION_COMPATIBLE TRUE)
                {% elif policy == "SameMajorVersion" %}
                if("{{ version }}" MATCHES "^([0-9]+)\\\\.")
                    set(CVF_VERSION_MAJOR {{ '${CMAKE_MATCH_1}' }})
                else()
                    set(CVF_VERSION_MAJOR "{{ version }}")
                endif()

                if(PACKAGE_FIND_VERSION_MAJOR STREQUAL CVF_VERSION_MAJOR)
                    set(PACKAGE_VERSION_COMPATIBLE TRUE)
                else()
                    set(PACKAGE_VERSION_COMPATIBLE FALSE)
                endif()
                {% elif policy == "SameMinorVersion" %}
                if("{{ version }}" MATCHES "^([0-9]+)\\.([0-9]+)")
                    set(CVF_VERSION_MAJOR "${CMAKE_MATCH_1}")
                    set(CVF_VERSION_MINOR "${CMAKE_MATCH_2}")
                else()
                    set(CVF_VERSION_MAJOR "{{ version }}")
                    set(CVF_VERSION_MINOR "")
                endif()
                if((PACKAGE_FIND_VERSION_MAJOR STREQUAL CVF_VERSION_MAJOR) AND
                    (PACKAGE_FIND_VERSION_MINOR STREQUAL CVF_VERSION_MINOR))
                  set(PACKAGE_VERSION_COMPATIBLE TRUE)
                else()
                  set(PACKAGE_VERSION_COMPATIBLE FALSE)
                endif()
                {% elif policy == "ExactVersion" %}
                if("{{ version }}" MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
                    set(CVF_VERSION_MAJOR "${CMAKE_MATCH_1}")
                    set(CVF_VERSION_MINOR "${CMAKE_MATCH_2}")
                    set(CVF_VERSION_MINOR "${CMAKE_MATCH_3}")
                else()
                    set(CVF_VERSION_MAJOR "{{ version }}")
                    set(CVF_VERSION_MINOR "")
                    set(CVF_VERSION_PATCH "")
                endif()
                if((PACKAGE_FIND_VERSION_MAJOR STREQUAL CVF_VERSION_MAJOR) AND
                    (PACKAGE_FIND_VERSION_MINOR STREQUAL CVF_VERSION_MINOR) AND
                    (PACKAGE_FIND_VERSION_PATCH STREQUAL CVF_VERSION_PATCH))
                  set(PACKAGE_VERSION_COMPATIBLE TRUE)
                else()
                  set(PACKAGE_VERSION_COMPATIBLE FALSE)
                endif()
                {% endif %}

                if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
                    set(PACKAGE_VERSION_EXACT TRUE)
                endif()
            endif()
            """)
        return ret
