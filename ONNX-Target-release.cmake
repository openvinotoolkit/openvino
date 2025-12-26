# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(onnx_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(onnx_FRAMEWORKS_FOUND_RELEASE "${onnx_FRAMEWORKS_RELEASE}" "${onnx_FRAMEWORK_DIRS_RELEASE}")

set(onnx_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET onnx_DEPS_TARGET)
    add_library(onnx_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET onnx_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${onnx_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${onnx_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:onnx_proto;protobuf::libprotobuf>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### onnx_DEPS_TARGET to all of them
conan_package_library_targets("${onnx_LIBS_RELEASE}"    # libraries
                              "${onnx_LIB_DIRS_RELEASE}" # package_libdir
                              "${onnx_BIN_DIRS_RELEASE}" # package_bindir
                              "${onnx_LIBRARY_TYPE_RELEASE}"
                              "${onnx_IS_HOST_WINDOWS_RELEASE}"
                              onnx_DEPS_TARGET
                              onnx_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "onnx"    # package_name
                              "${onnx_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${onnx_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT onnx #############

        set(onnx_onnx_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(onnx_onnx_FRAMEWORKS_FOUND_RELEASE "${onnx_onnx_FRAMEWORKS_RELEASE}" "${onnx_onnx_FRAMEWORK_DIRS_RELEASE}")

        set(onnx_onnx_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET onnx_onnx_DEPS_TARGET)
            add_library(onnx_onnx_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET onnx_onnx_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onnx_onnx_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${onnx_onnx_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${onnx_onnx_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'onnx_onnx_DEPS_TARGET' to all of them
        conan_package_library_targets("${onnx_onnx_LIBS_RELEASE}"
                              "${onnx_onnx_LIB_DIRS_RELEASE}"
                              "${onnx_onnx_BIN_DIRS_RELEASE}" # package_bindir
                              "${onnx_onnx_LIBRARY_TYPE_RELEASE}"
                              "${onnx_onnx_IS_HOST_WINDOWS_RELEASE}"
                              onnx_onnx_DEPS_TARGET
                              onnx_onnx_LIBRARIES_TARGETS
                              "_RELEASE"
                              "onnx_onnx"
                              "${onnx_onnx_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET onnx
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onnx_onnx_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${onnx_onnx_LIBRARIES_TARGETS}>
                     )

        if("${onnx_onnx_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET onnx
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         onnx_onnx_DEPS_TARGET)
        endif()

        set_property(TARGET onnx APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${onnx_onnx_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET onnx APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${onnx_onnx_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET onnx APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${onnx_onnx_LIB_DIRS_RELEASE}>)
        set_property(TARGET onnx APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${onnx_onnx_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET onnx APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${onnx_onnx_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT onnx_proto #############

        set(onnx_onnx_proto_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(onnx_onnx_proto_FRAMEWORKS_FOUND_RELEASE "${onnx_onnx_proto_FRAMEWORKS_RELEASE}" "${onnx_onnx_proto_FRAMEWORK_DIRS_RELEASE}")

        set(onnx_onnx_proto_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET onnx_onnx_proto_DEPS_TARGET)
            add_library(onnx_onnx_proto_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET onnx_onnx_proto_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onnx_onnx_proto_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${onnx_onnx_proto_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${onnx_onnx_proto_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'onnx_onnx_proto_DEPS_TARGET' to all of them
        conan_package_library_targets("${onnx_onnx_proto_LIBS_RELEASE}"
                              "${onnx_onnx_proto_LIB_DIRS_RELEASE}"
                              "${onnx_onnx_proto_BIN_DIRS_RELEASE}" # package_bindir
                              "${onnx_onnx_proto_LIBRARY_TYPE_RELEASE}"
                              "${onnx_onnx_proto_IS_HOST_WINDOWS_RELEASE}"
                              onnx_onnx_proto_DEPS_TARGET
                              onnx_onnx_proto_LIBRARIES_TARGETS
                              "_RELEASE"
                              "onnx_onnx_proto"
                              "${onnx_onnx_proto_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET onnx_proto
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onnx_onnx_proto_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${onnx_onnx_proto_LIBRARIES_TARGETS}>
                     )

        if("${onnx_onnx_proto_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET onnx_proto
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         onnx_onnx_proto_DEPS_TARGET)
        endif()

        set_property(TARGET onnx_proto APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${onnx_onnx_proto_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET onnx_proto APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${onnx_onnx_proto_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET onnx_proto APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${onnx_onnx_proto_LIB_DIRS_RELEASE}>)
        set_property(TARGET onnx_proto APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${onnx_onnx_proto_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET onnx_proto APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${onnx_onnx_proto_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET onnx::onnx APPEND PROPERTY INTERFACE_LINK_LIBRARIES onnx)
    set_property(TARGET onnx::onnx APPEND PROPERTY INTERFACE_LINK_LIBRARIES onnx_proto)

########## For the modules (FindXXX)
set(onnx_LIBRARIES_RELEASE onnx::onnx)
