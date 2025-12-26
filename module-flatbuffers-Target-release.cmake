# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(flatbuffers_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(flatbuffers_FRAMEWORKS_FOUND_RELEASE "${flatbuffers_FRAMEWORKS_RELEASE}" "${flatbuffers_FRAMEWORK_DIRS_RELEASE}")

set(flatbuffers_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET flatbuffers_DEPS_TARGET)
    add_library(flatbuffers_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET flatbuffers_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${flatbuffers_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${flatbuffers_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### flatbuffers_DEPS_TARGET to all of them
conan_package_library_targets("${flatbuffers_LIBS_RELEASE}"    # libraries
                              "${flatbuffers_LIB_DIRS_RELEASE}" # package_libdir
                              "${flatbuffers_BIN_DIRS_RELEASE}" # package_bindir
                              "${flatbuffers_LIBRARY_TYPE_RELEASE}"
                              "${flatbuffers_IS_HOST_WINDOWS_RELEASE}"
                              flatbuffers_DEPS_TARGET
                              flatbuffers_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "flatbuffers"    # package_name
                              "${flatbuffers_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${flatbuffers_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT flatbuffers::libflatbuffers #############

        set(flatbuffers_flatbuffers_libflatbuffers_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(flatbuffers_flatbuffers_libflatbuffers_FRAMEWORKS_FOUND_RELEASE "${flatbuffers_flatbuffers_libflatbuffers_FRAMEWORKS_RELEASE}" "${flatbuffers_flatbuffers_libflatbuffers_FRAMEWORK_DIRS_RELEASE}")

        set(flatbuffers_flatbuffers_libflatbuffers_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET flatbuffers_flatbuffers_libflatbuffers_DEPS_TARGET)
            add_library(flatbuffers_flatbuffers_libflatbuffers_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET flatbuffers_flatbuffers_libflatbuffers_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'flatbuffers_flatbuffers_libflatbuffers_DEPS_TARGET' to all of them
        conan_package_library_targets("${flatbuffers_flatbuffers_libflatbuffers_LIBS_RELEASE}"
                              "${flatbuffers_flatbuffers_libflatbuffers_LIB_DIRS_RELEASE}"
                              "${flatbuffers_flatbuffers_libflatbuffers_BIN_DIRS_RELEASE}" # package_bindir
                              "${flatbuffers_flatbuffers_libflatbuffers_LIBRARY_TYPE_RELEASE}"
                              "${flatbuffers_flatbuffers_libflatbuffers_IS_HOST_WINDOWS_RELEASE}"
                              flatbuffers_flatbuffers_libflatbuffers_DEPS_TARGET
                              flatbuffers_flatbuffers_libflatbuffers_LIBRARIES_TARGETS
                              "_RELEASE"
                              "flatbuffers_flatbuffers_libflatbuffers"
                              "${flatbuffers_flatbuffers_libflatbuffers_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET flatbuffers::libflatbuffers
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_LIBRARIES_TARGETS}>
                     )

        if("${flatbuffers_flatbuffers_libflatbuffers_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET flatbuffers::libflatbuffers
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         flatbuffers_flatbuffers_libflatbuffers_DEPS_TARGET)
        endif()

        set_property(TARGET flatbuffers::libflatbuffers APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET flatbuffers::libflatbuffers APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET flatbuffers::libflatbuffers APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_LIB_DIRS_RELEASE}>)
        set_property(TARGET flatbuffers::libflatbuffers APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET flatbuffers::libflatbuffers APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${flatbuffers_flatbuffers_libflatbuffers_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET flatbuffers::flatbuffers APPEND PROPERTY INTERFACE_LINK_LIBRARIES flatbuffers::libflatbuffers)

########## For the modules (FindXXX)
set(flatbuffers_LIBRARIES_RELEASE flatbuffers::flatbuffers)
