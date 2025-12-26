# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(opencl-clhpp-headers_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(opencl-clhpp-headers_FRAMEWORKS_FOUND_RELEASE "${opencl-clhpp-headers_FRAMEWORKS_RELEASE}" "${opencl-clhpp-headers_FRAMEWORK_DIRS_RELEASE}")

set(opencl-clhpp-headers_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET opencl-clhpp-headers_DEPS_TARGET)
    add_library(opencl-clhpp-headers_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET opencl-clhpp-headers_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${opencl-clhpp-headers_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${opencl-clhpp-headers_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:OpenCL::Headers>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### opencl-clhpp-headers_DEPS_TARGET to all of them
conan_package_library_targets("${opencl-clhpp-headers_LIBS_RELEASE}"    # libraries
                              "${opencl-clhpp-headers_LIB_DIRS_RELEASE}" # package_libdir
                              "${opencl-clhpp-headers_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencl-clhpp-headers_LIBRARY_TYPE_RELEASE}"
                              "${opencl-clhpp-headers_IS_HOST_WINDOWS_RELEASE}"
                              opencl-clhpp-headers_DEPS_TARGET
                              opencl-clhpp-headers_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "opencl-clhpp-headers"    # package_name
                              "${opencl-clhpp-headers_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${opencl-clhpp-headers_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET OpenCL::HeadersCpp
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_LIBRARIES_TARGETS}>
                 )

    if("${opencl-clhpp-headers_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET OpenCL::HeadersCpp
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     opencl-clhpp-headers_DEPS_TARGET)
    endif()

    set_property(TARGET OpenCL::HeadersCpp
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET OpenCL::HeadersCpp
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET OpenCL::HeadersCpp
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_LIB_DIRS_RELEASE}>)
    set_property(TARGET OpenCL::HeadersCpp
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET OpenCL::HeadersCpp
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${opencl-clhpp-headers_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(opencl-clhpp-headers_LIBRARIES_RELEASE OpenCL::HeadersCpp)
