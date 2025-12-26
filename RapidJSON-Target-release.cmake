# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(rapidjson_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(rapidjson_FRAMEWORKS_FOUND_RELEASE "${rapidjson_FRAMEWORKS_RELEASE}" "${rapidjson_FRAMEWORK_DIRS_RELEASE}")

set(rapidjson_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET rapidjson_DEPS_TARGET)
    add_library(rapidjson_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET rapidjson_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${rapidjson_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${rapidjson_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### rapidjson_DEPS_TARGET to all of them
conan_package_library_targets("${rapidjson_LIBS_RELEASE}"    # libraries
                              "${rapidjson_LIB_DIRS_RELEASE}" # package_libdir
                              "${rapidjson_BIN_DIRS_RELEASE}" # package_bindir
                              "${rapidjson_LIBRARY_TYPE_RELEASE}"
                              "${rapidjson_IS_HOST_WINDOWS_RELEASE}"
                              rapidjson_DEPS_TARGET
                              rapidjson_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "rapidjson"    # package_name
                              "${rapidjson_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${rapidjson_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET rapidjson
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${rapidjson_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${rapidjson_LIBRARIES_TARGETS}>
                 )

    if("${rapidjson_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET rapidjson
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     rapidjson_DEPS_TARGET)
    endif()

    set_property(TARGET rapidjson
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${rapidjson_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET rapidjson
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${rapidjson_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET rapidjson
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${rapidjson_LIB_DIRS_RELEASE}>)
    set_property(TARGET rapidjson
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${rapidjson_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET rapidjson
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${rapidjson_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(rapidjson_LIBRARIES_RELEASE rapidjson)
