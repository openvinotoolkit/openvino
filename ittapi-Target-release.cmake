# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(ittapi_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(ittapi_FRAMEWORKS_FOUND_RELEASE "${ittapi_FRAMEWORKS_RELEASE}" "${ittapi_FRAMEWORK_DIRS_RELEASE}")

set(ittapi_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET ittapi_DEPS_TARGET)
    add_library(ittapi_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET ittapi_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${ittapi_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${ittapi_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### ittapi_DEPS_TARGET to all of them
conan_package_library_targets("${ittapi_LIBS_RELEASE}"    # libraries
                              "${ittapi_LIB_DIRS_RELEASE}" # package_libdir
                              "${ittapi_BIN_DIRS_RELEASE}" # package_bindir
                              "${ittapi_LIBRARY_TYPE_RELEASE}"
                              "${ittapi_IS_HOST_WINDOWS_RELEASE}"
                              ittapi_DEPS_TARGET
                              ittapi_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "ittapi"    # package_name
                              "${ittapi_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${ittapi_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET ittapi::ittapi
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${ittapi_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${ittapi_LIBRARIES_TARGETS}>
                 )

    if("${ittapi_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET ittapi::ittapi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     ittapi_DEPS_TARGET)
    endif()

    set_property(TARGET ittapi::ittapi
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${ittapi_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET ittapi::ittapi
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${ittapi_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET ittapi::ittapi
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${ittapi_LIB_DIRS_RELEASE}>)
    set_property(TARGET ittapi::ittapi
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${ittapi_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET ittapi::ittapi
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${ittapi_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(ittapi_LIBRARIES_RELEASE ittapi::ittapi)
