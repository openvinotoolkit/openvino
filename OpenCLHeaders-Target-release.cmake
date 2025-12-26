# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(opencl-headers_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(opencl-headers_FRAMEWORKS_FOUND_RELEASE "${opencl-headers_FRAMEWORKS_RELEASE}" "${opencl-headers_FRAMEWORK_DIRS_RELEASE}")

set(opencl-headers_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET opencl-headers_DEPS_TARGET)
    add_library(opencl-headers_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET opencl-headers_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${opencl-headers_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${opencl-headers_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### opencl-headers_DEPS_TARGET to all of them
conan_package_library_targets("${opencl-headers_LIBS_RELEASE}"    # libraries
                              "${opencl-headers_LIB_DIRS_RELEASE}" # package_libdir
                              "${opencl-headers_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencl-headers_LIBRARY_TYPE_RELEASE}"
                              "${opencl-headers_IS_HOST_WINDOWS_RELEASE}"
                              opencl-headers_DEPS_TARGET
                              opencl-headers_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "opencl-headers"    # package_name
                              "${opencl-headers_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${opencl-headers_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET OpenCL::Headers
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${opencl-headers_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${opencl-headers_LIBRARIES_TARGETS}>
                 )

    if("${opencl-headers_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET OpenCL::Headers
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     opencl-headers_DEPS_TARGET)
    endif()

    set_property(TARGET OpenCL::Headers
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${opencl-headers_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET OpenCL::Headers
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${opencl-headers_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET OpenCL::Headers
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${opencl-headers_LIB_DIRS_RELEASE}>)
    set_property(TARGET OpenCL::Headers
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${opencl-headers_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET OpenCL::Headers
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${opencl-headers_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(opencl-headers_LIBRARIES_RELEASE OpenCL::Headers)
