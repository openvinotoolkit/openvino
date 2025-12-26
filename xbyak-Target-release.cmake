# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(xbyak_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(xbyak_FRAMEWORKS_FOUND_RELEASE "${xbyak_FRAMEWORKS_RELEASE}" "${xbyak_FRAMEWORK_DIRS_RELEASE}")

set(xbyak_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET xbyak_DEPS_TARGET)
    add_library(xbyak_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET xbyak_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${xbyak_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${xbyak_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### xbyak_DEPS_TARGET to all of them
conan_package_library_targets("${xbyak_LIBS_RELEASE}"    # libraries
                              "${xbyak_LIB_DIRS_RELEASE}" # package_libdir
                              "${xbyak_BIN_DIRS_RELEASE}" # package_bindir
                              "${xbyak_LIBRARY_TYPE_RELEASE}"
                              "${xbyak_IS_HOST_WINDOWS_RELEASE}"
                              xbyak_DEPS_TARGET
                              xbyak_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "xbyak"    # package_name
                              "${xbyak_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${xbyak_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET xbyak::xbyak
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${xbyak_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${xbyak_LIBRARIES_TARGETS}>
                 )

    if("${xbyak_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET xbyak::xbyak
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     xbyak_DEPS_TARGET)
    endif()

    set_property(TARGET xbyak::xbyak
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${xbyak_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET xbyak::xbyak
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${xbyak_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET xbyak::xbyak
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${xbyak_LIB_DIRS_RELEASE}>)
    set_property(TARGET xbyak::xbyak
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${xbyak_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET xbyak::xbyak
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${xbyak_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(xbyak_LIBRARIES_RELEASE xbyak::xbyak)
