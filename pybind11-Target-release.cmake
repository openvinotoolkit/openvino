# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(pybind11_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(pybind11_FRAMEWORKS_FOUND_RELEASE "${pybind11_FRAMEWORKS_RELEASE}" "${pybind11_FRAMEWORK_DIRS_RELEASE}")

set(pybind11_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET pybind11_DEPS_TARGET)
    add_library(pybind11_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET pybind11_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${pybind11_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${pybind11_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:pybind11::headers;pybind11::pybind11>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### pybind11_DEPS_TARGET to all of them
conan_package_library_targets("${pybind11_LIBS_RELEASE}"    # libraries
                              "${pybind11_LIB_DIRS_RELEASE}" # package_libdir
                              "${pybind11_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_DEPS_TARGET
                              pybind11_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "pybind11"    # package_name
                              "${pybind11_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${pybind11_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT pybind11::python2_no_register #############

        set(pybind11_pybind11_python2_no_register_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_python2_no_register_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_python2_no_register_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_python2_no_register_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_python2_no_register_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_python2_no_register_DEPS_TARGET)
            add_library(pybind11_pybind11_python2_no_register_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_python2_no_register_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_python2_no_register_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_python2_no_register_LIBS_RELEASE}"
                              "${pybind11_pybind11_python2_no_register_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_python2_no_register_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_python2_no_register_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_python2_no_register_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_python2_no_register_DEPS_TARGET
                              pybind11_pybind11_python2_no_register_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_python2_no_register"
                              "${pybind11_pybind11_python2_no_register_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::python2_no_register
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_python2_no_register_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::python2_no_register
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_python2_no_register_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::python2_no_register APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::python2_no_register APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::python2_no_register APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::python2_no_register APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::python2_no_register APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_python2_no_register_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::opt_size #############

        set(pybind11_pybind11_opt_size_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_opt_size_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_opt_size_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_opt_size_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_opt_size_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_opt_size_DEPS_TARGET)
            add_library(pybind11_pybind11_opt_size_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_opt_size_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_opt_size_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_opt_size_LIBS_RELEASE}"
                              "${pybind11_pybind11_opt_size_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_opt_size_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_opt_size_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_opt_size_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_opt_size_DEPS_TARGET
                              pybind11_pybind11_opt_size_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_opt_size"
                              "${pybind11_pybind11_opt_size_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::opt_size
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_opt_size_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::opt_size
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_opt_size_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::opt_size APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::opt_size APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::opt_size APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::opt_size APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::opt_size APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_opt_size_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::thin_lto #############

        set(pybind11_pybind11_thin_lto_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_thin_lto_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_thin_lto_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_thin_lto_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_thin_lto_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_thin_lto_DEPS_TARGET)
            add_library(pybind11_pybind11_thin_lto_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_thin_lto_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_thin_lto_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_thin_lto_LIBS_RELEASE}"
                              "${pybind11_pybind11_thin_lto_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_thin_lto_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_thin_lto_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_thin_lto_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_thin_lto_DEPS_TARGET
                              pybind11_pybind11_thin_lto_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_thin_lto"
                              "${pybind11_pybind11_thin_lto_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::thin_lto
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_thin_lto_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::thin_lto
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_thin_lto_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::thin_lto APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::thin_lto APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::thin_lto APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::thin_lto APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::thin_lto APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_thin_lto_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::lto #############

        set(pybind11_pybind11_lto_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_lto_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_lto_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_lto_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_lto_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_lto_DEPS_TARGET)
            add_library(pybind11_pybind11_lto_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_lto_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_lto_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_lto_LIBS_RELEASE}"
                              "${pybind11_pybind11_lto_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_lto_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_lto_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_lto_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_lto_DEPS_TARGET
                              pybind11_pybind11_lto_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_lto"
                              "${pybind11_pybind11_lto_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::lto
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_lto_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::lto
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_lto_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::lto APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::lto APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::lto APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::lto APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::lto APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_lto_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::windows_extras #############

        set(pybind11_pybind11_windows_extras_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_windows_extras_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_windows_extras_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_windows_extras_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_windows_extras_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_windows_extras_DEPS_TARGET)
            add_library(pybind11_pybind11_windows_extras_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_windows_extras_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_windows_extras_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_windows_extras_LIBS_RELEASE}"
                              "${pybind11_pybind11_windows_extras_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_windows_extras_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_windows_extras_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_windows_extras_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_windows_extras_DEPS_TARGET
                              pybind11_pybind11_windows_extras_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_windows_extras"
                              "${pybind11_pybind11_windows_extras_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::windows_extras
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_windows_extras_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::windows_extras
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_windows_extras_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::windows_extras APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::windows_extras APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::windows_extras APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::windows_extras APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::windows_extras APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_windows_extras_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::python_link_helper #############

        set(pybind11_pybind11_python_link_helper_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_python_link_helper_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_python_link_helper_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_python_link_helper_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_python_link_helper_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_python_link_helper_DEPS_TARGET)
            add_library(pybind11_pybind11_python_link_helper_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_python_link_helper_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_python_link_helper_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_python_link_helper_LIBS_RELEASE}"
                              "${pybind11_pybind11_python_link_helper_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_python_link_helper_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_python_link_helper_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_python_link_helper_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_python_link_helper_DEPS_TARGET
                              pybind11_pybind11_python_link_helper_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_python_link_helper"
                              "${pybind11_pybind11_python_link_helper_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::python_link_helper
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_python_link_helper_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::python_link_helper
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_python_link_helper_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::python_link_helper APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::python_link_helper APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::python_link_helper APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::python_link_helper APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::python_link_helper APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_python_link_helper_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::module #############

        set(pybind11_pybind11_module_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_module_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_module_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_module_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_module_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_module_DEPS_TARGET)
            add_library(pybind11_pybind11_module_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_module_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_module_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_module_LIBS_RELEASE}"
                              "${pybind11_pybind11_module_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_module_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_module_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_module_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_module_DEPS_TARGET
                              pybind11_pybind11_module_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_module"
                              "${pybind11_pybind11_module_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::module
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_module_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::module
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_module_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::module APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::module APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::module APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::module APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::module APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_module_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::embed #############

        set(pybind11_pybind11_embed_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_embed_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_embed_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_embed_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_embed_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_embed_DEPS_TARGET)
            add_library(pybind11_pybind11_embed_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_embed_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_embed_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_embed_LIBS_RELEASE}"
                              "${pybind11_pybind11_embed_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_embed_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_embed_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_embed_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_embed_DEPS_TARGET
                              pybind11_pybind11_embed_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_embed"
                              "${pybind11_pybind11_embed_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::embed
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_embed_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::embed
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_embed_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::embed APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::embed APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::embed APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::embed APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::embed APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_embed_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::pybind11 #############

        set(pybind11_pybind11_pybind11_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_pybind11_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_pybind11_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_pybind11_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_pybind11_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_pybind11_DEPS_TARGET)
            add_library(pybind11_pybind11_pybind11_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_pybind11_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_pybind11_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_pybind11_LIBS_RELEASE}"
                              "${pybind11_pybind11_pybind11_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_pybind11_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_pybind11_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_pybind11_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_pybind11_DEPS_TARGET
                              pybind11_pybind11_pybind11_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_pybind11"
                              "${pybind11_pybind11_pybind11_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::pybind11
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_pybind11_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::pybind11
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_pybind11_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::pybind11 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::pybind11 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::pybind11 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::pybind11 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::pybind11 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_pybind11_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT pybind11::headers #############

        set(pybind11_pybind11_headers_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(pybind11_pybind11_headers_FRAMEWORKS_FOUND_RELEASE "${pybind11_pybind11_headers_FRAMEWORKS_RELEASE}" "${pybind11_pybind11_headers_FRAMEWORK_DIRS_RELEASE}")

        set(pybind11_pybind11_headers_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET pybind11_pybind11_headers_DEPS_TARGET)
            add_library(pybind11_pybind11_headers_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET pybind11_pybind11_headers_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'pybind11_pybind11_headers_DEPS_TARGET' to all of them
        conan_package_library_targets("${pybind11_pybind11_headers_LIBS_RELEASE}"
                              "${pybind11_pybind11_headers_LIB_DIRS_RELEASE}"
                              "${pybind11_pybind11_headers_BIN_DIRS_RELEASE}" # package_bindir
                              "${pybind11_pybind11_headers_LIBRARY_TYPE_RELEASE}"
                              "${pybind11_pybind11_headers_IS_HOST_WINDOWS_RELEASE}"
                              pybind11_pybind11_headers_DEPS_TARGET
                              pybind11_pybind11_headers_LIBRARIES_TARGETS
                              "_RELEASE"
                              "pybind11_pybind11_headers"
                              "${pybind11_pybind11_headers_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET pybind11::headers
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_LIBRARIES_TARGETS}>
                     )

        if("${pybind11_pybind11_headers_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET pybind11::headers
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         pybind11_pybind11_headers_DEPS_TARGET)
        endif()

        set_property(TARGET pybind11::headers APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET pybind11::headers APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET pybind11::headers APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_LIB_DIRS_RELEASE}>)
        set_property(TARGET pybind11::headers APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET pybind11::headers APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${pybind11_pybind11_headers_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python2_no_register)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::opt_size)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::thin_lto)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::lto)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::windows_extras)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python_link_helper)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::module)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::embed)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::pybind11)
    set_property(TARGET pybind11_all_do_not_use APPEND PROPERTY INTERFACE_LINK_LIBRARIES pybind11::headers)

########## For the modules (FindXXX)
set(pybind11_LIBRARIES_RELEASE pybind11_all_do_not_use)
