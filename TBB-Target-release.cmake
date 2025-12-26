# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(onetbb_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(onetbb_FRAMEWORKS_FOUND_RELEASE "${onetbb_FRAMEWORKS_RELEASE}" "${onetbb_FRAMEWORK_DIRS_RELEASE}")

set(onetbb_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET onetbb_DEPS_TARGET)
    add_library(onetbb_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET onetbb_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${onetbb_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${onetbb_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:TBB::tbbmalloc>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### onetbb_DEPS_TARGET to all of them
conan_package_library_targets("${onetbb_LIBS_RELEASE}"    # libraries
                              "${onetbb_LIB_DIRS_RELEASE}" # package_libdir
                              "${onetbb_BIN_DIRS_RELEASE}" # package_bindir
                              "${onetbb_LIBRARY_TYPE_RELEASE}"
                              "${onetbb_IS_HOST_WINDOWS_RELEASE}"
                              onetbb_DEPS_TARGET
                              onetbb_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "onetbb"    # package_name
                              "${onetbb_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${onetbb_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT TBB::tbbmalloc_proxy #############

        set(onetbb_TBB_tbbmalloc_proxy_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(onetbb_TBB_tbbmalloc_proxy_FRAMEWORKS_FOUND_RELEASE "${onetbb_TBB_tbbmalloc_proxy_FRAMEWORKS_RELEASE}" "${onetbb_TBB_tbbmalloc_proxy_FRAMEWORK_DIRS_RELEASE}")

        set(onetbb_TBB_tbbmalloc_proxy_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET onetbb_TBB_tbbmalloc_proxy_DEPS_TARGET)
            add_library(onetbb_TBB_tbbmalloc_proxy_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET onetbb_TBB_tbbmalloc_proxy_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'onetbb_TBB_tbbmalloc_proxy_DEPS_TARGET' to all of them
        conan_package_library_targets("${onetbb_TBB_tbbmalloc_proxy_LIBS_RELEASE}"
                              "${onetbb_TBB_tbbmalloc_proxy_LIB_DIRS_RELEASE}"
                              "${onetbb_TBB_tbbmalloc_proxy_BIN_DIRS_RELEASE}" # package_bindir
                              "${onetbb_TBB_tbbmalloc_proxy_LIBRARY_TYPE_RELEASE}"
                              "${onetbb_TBB_tbbmalloc_proxy_IS_HOST_WINDOWS_RELEASE}"
                              onetbb_TBB_tbbmalloc_proxy_DEPS_TARGET
                              onetbb_TBB_tbbmalloc_proxy_LIBRARIES_TARGETS
                              "_RELEASE"
                              "onetbb_TBB_tbbmalloc_proxy"
                              "${onetbb_TBB_tbbmalloc_proxy_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET TBB::tbbmalloc_proxy
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_LIBRARIES_TARGETS}>
                     )

        if("${onetbb_TBB_tbbmalloc_proxy_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET TBB::tbbmalloc_proxy
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         onetbb_TBB_tbbmalloc_proxy_DEPS_TARGET)
        endif()

        set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_LIB_DIRS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_proxy_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT TBB::tbbmalloc #############

        set(onetbb_TBB_tbbmalloc_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(onetbb_TBB_tbbmalloc_FRAMEWORKS_FOUND_RELEASE "${onetbb_TBB_tbbmalloc_FRAMEWORKS_RELEASE}" "${onetbb_TBB_tbbmalloc_FRAMEWORK_DIRS_RELEASE}")

        set(onetbb_TBB_tbbmalloc_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET onetbb_TBB_tbbmalloc_DEPS_TARGET)
            add_library(onetbb_TBB_tbbmalloc_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET onetbb_TBB_tbbmalloc_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'onetbb_TBB_tbbmalloc_DEPS_TARGET' to all of them
        conan_package_library_targets("${onetbb_TBB_tbbmalloc_LIBS_RELEASE}"
                              "${onetbb_TBB_tbbmalloc_LIB_DIRS_RELEASE}"
                              "${onetbb_TBB_tbbmalloc_BIN_DIRS_RELEASE}" # package_bindir
                              "${onetbb_TBB_tbbmalloc_LIBRARY_TYPE_RELEASE}"
                              "${onetbb_TBB_tbbmalloc_IS_HOST_WINDOWS_RELEASE}"
                              onetbb_TBB_tbbmalloc_DEPS_TARGET
                              onetbb_TBB_tbbmalloc_LIBRARIES_TARGETS
                              "_RELEASE"
                              "onetbb_TBB_tbbmalloc"
                              "${onetbb_TBB_tbbmalloc_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET TBB::tbbmalloc
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_LIBRARIES_TARGETS}>
                     )

        if("${onetbb_TBB_tbbmalloc_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET TBB::tbbmalloc
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         onetbb_TBB_tbbmalloc_DEPS_TARGET)
        endif()

        set_property(TARGET TBB::tbbmalloc APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_LIB_DIRS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET TBB::tbbmalloc APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbbmalloc_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT TBB::tbb #############

        set(onetbb_TBB_tbb_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(onetbb_TBB_tbb_FRAMEWORKS_FOUND_RELEASE "${onetbb_TBB_tbb_FRAMEWORKS_RELEASE}" "${onetbb_TBB_tbb_FRAMEWORK_DIRS_RELEASE}")

        set(onetbb_TBB_tbb_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET onetbb_TBB_tbb_DEPS_TARGET)
            add_library(onetbb_TBB_tbb_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET onetbb_TBB_tbb_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'onetbb_TBB_tbb_DEPS_TARGET' to all of them
        conan_package_library_targets("${onetbb_TBB_tbb_LIBS_RELEASE}"
                              "${onetbb_TBB_tbb_LIB_DIRS_RELEASE}"
                              "${onetbb_TBB_tbb_BIN_DIRS_RELEASE}" # package_bindir
                              "${onetbb_TBB_tbb_LIBRARY_TYPE_RELEASE}"
                              "${onetbb_TBB_tbb_IS_HOST_WINDOWS_RELEASE}"
                              onetbb_TBB_tbb_DEPS_TARGET
                              onetbb_TBB_tbb_LIBRARIES_TARGETS
                              "_RELEASE"
                              "onetbb_TBB_tbb"
                              "${onetbb_TBB_tbb_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET TBB::tbb
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_LIBRARIES_TARGETS}>
                     )

        if("${onetbb_TBB_tbb_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET TBB::tbb
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         onetbb_TBB_tbb_DEPS_TARGET)
        endif()

        set_property(TARGET TBB::tbb APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET TBB::tbb APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET TBB::tbb APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_LIB_DIRS_RELEASE}>)
        set_property(TARGET TBB::tbb APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET TBB::tbb APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${onetbb_TBB_tbb_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET onetbb::onetbb APPEND PROPERTY INTERFACE_LINK_LIBRARIES TBB::tbbmalloc_proxy)
    set_property(TARGET onetbb::onetbb APPEND PROPERTY INTERFACE_LINK_LIBRARIES TBB::tbbmalloc)
    set_property(TARGET onetbb::onetbb APPEND PROPERTY INTERFACE_LINK_LIBRARIES TBB::tbb)

########## For the modules (FindXXX)
set(onetbb_LIBRARIES_RELEASE onetbb::onetbb)
