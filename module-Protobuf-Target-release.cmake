# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(protobuf_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(protobuf_FRAMEWORKS_FOUND_RELEASE "${protobuf_FRAMEWORKS_RELEASE}" "${protobuf_FRAMEWORK_DIRS_RELEASE}")

set(protobuf_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET protobuf_DEPS_TARGET)
    add_library(protobuf_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET protobuf_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${protobuf_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${protobuf_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:protobuf::libprotobuf>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### protobuf_DEPS_TARGET to all of them
conan_package_library_targets("${protobuf_LIBS_RELEASE}"    # libraries
                              "${protobuf_LIB_DIRS_RELEASE}" # package_libdir
                              "${protobuf_BIN_DIRS_RELEASE}" # package_bindir
                              "${protobuf_LIBRARY_TYPE_RELEASE}"
                              "${protobuf_IS_HOST_WINDOWS_RELEASE}"
                              protobuf_DEPS_TARGET
                              protobuf_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "protobuf"    # package_name
                              "${protobuf_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${protobuf_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT protobuf::libprotoc #############

        set(protobuf_protobuf_libprotoc_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(protobuf_protobuf_libprotoc_FRAMEWORKS_FOUND_RELEASE "${protobuf_protobuf_libprotoc_FRAMEWORKS_RELEASE}" "${protobuf_protobuf_libprotoc_FRAMEWORK_DIRS_RELEASE}")

        set(protobuf_protobuf_libprotoc_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET protobuf_protobuf_libprotoc_DEPS_TARGET)
            add_library(protobuf_protobuf_libprotoc_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET protobuf_protobuf_libprotoc_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'protobuf_protobuf_libprotoc_DEPS_TARGET' to all of them
        conan_package_library_targets("${protobuf_protobuf_libprotoc_LIBS_RELEASE}"
                              "${protobuf_protobuf_libprotoc_LIB_DIRS_RELEASE}"
                              "${protobuf_protobuf_libprotoc_BIN_DIRS_RELEASE}" # package_bindir
                              "${protobuf_protobuf_libprotoc_LIBRARY_TYPE_RELEASE}"
                              "${protobuf_protobuf_libprotoc_IS_HOST_WINDOWS_RELEASE}"
                              protobuf_protobuf_libprotoc_DEPS_TARGET
                              protobuf_protobuf_libprotoc_LIBRARIES_TARGETS
                              "_RELEASE"
                              "protobuf_protobuf_libprotoc"
                              "${protobuf_protobuf_libprotoc_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET protobuf::libprotoc
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_LIBRARIES_TARGETS}>
                     )

        if("${protobuf_protobuf_libprotoc_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET protobuf::libprotoc
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         protobuf_protobuf_libprotoc_DEPS_TARGET)
        endif()

        set_property(TARGET protobuf::libprotoc APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET protobuf::libprotoc APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET protobuf::libprotoc APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_LIB_DIRS_RELEASE}>)
        set_property(TARGET protobuf::libprotoc APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET protobuf::libprotoc APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotoc_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT protobuf::libprotobuf-lite #############

        set(protobuf_protobuf_libprotobuf-lite_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(protobuf_protobuf_libprotobuf-lite_FRAMEWORKS_FOUND_RELEASE "${protobuf_protobuf_libprotobuf-lite_FRAMEWORKS_RELEASE}" "${protobuf_protobuf_libprotobuf-lite_FRAMEWORK_DIRS_RELEASE}")

        set(protobuf_protobuf_libprotobuf-lite_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET protobuf_protobuf_libprotobuf-lite_DEPS_TARGET)
            add_library(protobuf_protobuf_libprotobuf-lite_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET protobuf_protobuf_libprotobuf-lite_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'protobuf_protobuf_libprotobuf-lite_DEPS_TARGET' to all of them
        conan_package_library_targets("${protobuf_protobuf_libprotobuf-lite_LIBS_RELEASE}"
                              "${protobuf_protobuf_libprotobuf-lite_LIB_DIRS_RELEASE}"
                              "${protobuf_protobuf_libprotobuf-lite_BIN_DIRS_RELEASE}" # package_bindir
                              "${protobuf_protobuf_libprotobuf-lite_LIBRARY_TYPE_RELEASE}"
                              "${protobuf_protobuf_libprotobuf-lite_IS_HOST_WINDOWS_RELEASE}"
                              protobuf_protobuf_libprotobuf-lite_DEPS_TARGET
                              protobuf_protobuf_libprotobuf-lite_LIBRARIES_TARGETS
                              "_RELEASE"
                              "protobuf_protobuf_libprotobuf-lite"
                              "${protobuf_protobuf_libprotobuf-lite_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET protobuf::libprotobuf-lite
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_LIBRARIES_TARGETS}>
                     )

        if("${protobuf_protobuf_libprotobuf-lite_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET protobuf::libprotobuf-lite
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         protobuf_protobuf_libprotobuf-lite_DEPS_TARGET)
        endif()

        set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_LIB_DIRS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf-lite_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT protobuf::libprotobuf #############

        set(protobuf_protobuf_libprotobuf_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(protobuf_protobuf_libprotobuf_FRAMEWORKS_FOUND_RELEASE "${protobuf_protobuf_libprotobuf_FRAMEWORKS_RELEASE}" "${protobuf_protobuf_libprotobuf_FRAMEWORK_DIRS_RELEASE}")

        set(protobuf_protobuf_libprotobuf_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET protobuf_protobuf_libprotobuf_DEPS_TARGET)
            add_library(protobuf_protobuf_libprotobuf_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET protobuf_protobuf_libprotobuf_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'protobuf_protobuf_libprotobuf_DEPS_TARGET' to all of them
        conan_package_library_targets("${protobuf_protobuf_libprotobuf_LIBS_RELEASE}"
                              "${protobuf_protobuf_libprotobuf_LIB_DIRS_RELEASE}"
                              "${protobuf_protobuf_libprotobuf_BIN_DIRS_RELEASE}" # package_bindir
                              "${protobuf_protobuf_libprotobuf_LIBRARY_TYPE_RELEASE}"
                              "${protobuf_protobuf_libprotobuf_IS_HOST_WINDOWS_RELEASE}"
                              protobuf_protobuf_libprotobuf_DEPS_TARGET
                              protobuf_protobuf_libprotobuf_LIBRARIES_TARGETS
                              "_RELEASE"
                              "protobuf_protobuf_libprotobuf"
                              "${protobuf_protobuf_libprotobuf_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET protobuf::libprotobuf
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_LIBRARIES_TARGETS}>
                     )

        if("${protobuf_protobuf_libprotobuf_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET protobuf::libprotobuf
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         protobuf_protobuf_libprotobuf_DEPS_TARGET)
        endif()

        set_property(TARGET protobuf::libprotobuf APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_LIB_DIRS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET protobuf::libprotobuf APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${protobuf_protobuf_libprotobuf_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET protobuf::protobuf APPEND PROPERTY INTERFACE_LINK_LIBRARIES protobuf::libprotoc)
    set_property(TARGET protobuf::protobuf APPEND PROPERTY INTERFACE_LINK_LIBRARIES protobuf::libprotobuf-lite)
    set_property(TARGET protobuf::protobuf APPEND PROPERTY INTERFACE_LINK_LIBRARIES protobuf::libprotobuf)

########## For the modules (FindXXX)
set(protobuf_LIBRARIES_RELEASE protobuf::protobuf)
