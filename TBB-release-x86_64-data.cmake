########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND onetbb_COMPONENT_NAMES TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
list(REMOVE_DUPLICATES onetbb_COMPONENT_NAMES)
if(DEFINED onetbb_FIND_DEPENDENCY_NAMES)
  list(APPEND onetbb_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES onetbb_FIND_DEPENDENCY_NAMES)
else()
  set(onetbb_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(onetbb_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/onetb3727ded68a9f8/p")
set(onetbb_BUILD_MODULES_PATHS_RELEASE )


set(onetbb_INCLUDE_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/include")
set(onetbb_RES_DIRS_RELEASE )
set(onetbb_DEFINITIONS_RELEASE )
set(onetbb_SHARED_LINK_FLAGS_RELEASE )
set(onetbb_EXE_LINK_FLAGS_RELEASE )
set(onetbb_OBJECTS_RELEASE )
set(onetbb_COMPILE_DEFINITIONS_RELEASE )
set(onetbb_COMPILE_OPTIONS_C_RELEASE )
set(onetbb_COMPILE_OPTIONS_CXX_RELEASE )
set(onetbb_LIB_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/lib")
set(onetbb_BIN_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/bin")
set(onetbb_LIBRARY_TYPE_RELEASE SHARED)
set(onetbb_IS_HOST_WINDOWS_RELEASE 0)
set(onetbb_LIBS_RELEASE tbbmalloc_proxy tbbmalloc tbb)
set(onetbb_SYSTEM_LIBS_RELEASE m dl pthread rt)
set(onetbb_FRAMEWORK_DIRS_RELEASE )
set(onetbb_FRAMEWORKS_RELEASE )
set(onetbb_BUILD_DIRS_RELEASE )
set(onetbb_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(onetbb_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onetbb_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onetbb_COMPILE_OPTIONS_C_RELEASE}>")
set(onetbb_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onetbb_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onetbb_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onetbb_EXE_LINK_FLAGS_RELEASE}>")


set(onetbb_COMPONENTS_RELEASE TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
########### COMPONENT TBB::tbbmalloc_proxy VARIABLES ############################################

set(onetbb_TBB_tbbmalloc_proxy_INCLUDE_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/include")
set(onetbb_TBB_tbbmalloc_proxy_LIB_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/lib")
set(onetbb_TBB_tbbmalloc_proxy_BIN_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/bin")
set(onetbb_TBB_tbbmalloc_proxy_LIBRARY_TYPE_RELEASE SHARED)
set(onetbb_TBB_tbbmalloc_proxy_IS_HOST_WINDOWS_RELEASE 0)
set(onetbb_TBB_tbbmalloc_proxy_RES_DIRS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_DEFINITIONS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_OBJECTS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_COMPILE_DEFINITIONS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_COMPILE_OPTIONS_C_RELEASE "")
set(onetbb_TBB_tbbmalloc_proxy_COMPILE_OPTIONS_CXX_RELEASE "")
set(onetbb_TBB_tbbmalloc_proxy_LIBS_RELEASE tbbmalloc_proxy)
set(onetbb_TBB_tbbmalloc_proxy_SYSTEM_LIBS_RELEASE m dl pthread)
set(onetbb_TBB_tbbmalloc_proxy_FRAMEWORK_DIRS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_FRAMEWORKS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_DEPENDENCIES_RELEASE TBB::tbbmalloc)
set(onetbb_TBB_tbbmalloc_proxy_SHARED_LINK_FLAGS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_EXE_LINK_FLAGS_RELEASE )
set(onetbb_TBB_tbbmalloc_proxy_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(onetbb_TBB_tbbmalloc_proxy_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onetbb_TBB_tbbmalloc_proxy_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onetbb_TBB_tbbmalloc_proxy_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onetbb_TBB_tbbmalloc_proxy_EXE_LINK_FLAGS_RELEASE}>
)
set(onetbb_TBB_tbbmalloc_proxy_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onetbb_TBB_tbbmalloc_proxy_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onetbb_TBB_tbbmalloc_proxy_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT TBB::tbbmalloc VARIABLES ############################################

set(onetbb_TBB_tbbmalloc_INCLUDE_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/include")
set(onetbb_TBB_tbbmalloc_LIB_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/lib")
set(onetbb_TBB_tbbmalloc_BIN_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/bin")
set(onetbb_TBB_tbbmalloc_LIBRARY_TYPE_RELEASE SHARED)
set(onetbb_TBB_tbbmalloc_IS_HOST_WINDOWS_RELEASE 0)
set(onetbb_TBB_tbbmalloc_RES_DIRS_RELEASE )
set(onetbb_TBB_tbbmalloc_DEFINITIONS_RELEASE )
set(onetbb_TBB_tbbmalloc_OBJECTS_RELEASE )
set(onetbb_TBB_tbbmalloc_COMPILE_DEFINITIONS_RELEASE )
set(onetbb_TBB_tbbmalloc_COMPILE_OPTIONS_C_RELEASE "")
set(onetbb_TBB_tbbmalloc_COMPILE_OPTIONS_CXX_RELEASE "")
set(onetbb_TBB_tbbmalloc_LIBS_RELEASE tbbmalloc)
set(onetbb_TBB_tbbmalloc_SYSTEM_LIBS_RELEASE dl pthread)
set(onetbb_TBB_tbbmalloc_FRAMEWORK_DIRS_RELEASE )
set(onetbb_TBB_tbbmalloc_FRAMEWORKS_RELEASE )
set(onetbb_TBB_tbbmalloc_DEPENDENCIES_RELEASE )
set(onetbb_TBB_tbbmalloc_SHARED_LINK_FLAGS_RELEASE )
set(onetbb_TBB_tbbmalloc_EXE_LINK_FLAGS_RELEASE )
set(onetbb_TBB_tbbmalloc_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(onetbb_TBB_tbbmalloc_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onetbb_TBB_tbbmalloc_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onetbb_TBB_tbbmalloc_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onetbb_TBB_tbbmalloc_EXE_LINK_FLAGS_RELEASE}>
)
set(onetbb_TBB_tbbmalloc_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onetbb_TBB_tbbmalloc_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onetbb_TBB_tbbmalloc_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT TBB::tbb VARIABLES ############################################

set(onetbb_TBB_tbb_INCLUDE_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/include")
set(onetbb_TBB_tbb_LIB_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/lib")
set(onetbb_TBB_tbb_BIN_DIRS_RELEASE "${onetbb_PACKAGE_FOLDER_RELEASE}/bin")
set(onetbb_TBB_tbb_LIBRARY_TYPE_RELEASE SHARED)
set(onetbb_TBB_tbb_IS_HOST_WINDOWS_RELEASE 0)
set(onetbb_TBB_tbb_RES_DIRS_RELEASE )
set(onetbb_TBB_tbb_DEFINITIONS_RELEASE )
set(onetbb_TBB_tbb_OBJECTS_RELEASE )
set(onetbb_TBB_tbb_COMPILE_DEFINITIONS_RELEASE )
set(onetbb_TBB_tbb_COMPILE_OPTIONS_C_RELEASE "")
set(onetbb_TBB_tbb_COMPILE_OPTIONS_CXX_RELEASE "")
set(onetbb_TBB_tbb_LIBS_RELEASE tbb)
set(onetbb_TBB_tbb_SYSTEM_LIBS_RELEASE m dl rt pthread)
set(onetbb_TBB_tbb_FRAMEWORK_DIRS_RELEASE )
set(onetbb_TBB_tbb_FRAMEWORKS_RELEASE )
set(onetbb_TBB_tbb_DEPENDENCIES_RELEASE )
set(onetbb_TBB_tbb_SHARED_LINK_FLAGS_RELEASE )
set(onetbb_TBB_tbb_EXE_LINK_FLAGS_RELEASE )
set(onetbb_TBB_tbb_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(onetbb_TBB_tbb_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onetbb_TBB_tbb_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onetbb_TBB_tbb_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onetbb_TBB_tbb_EXE_LINK_FLAGS_RELEASE}>
)
set(onetbb_TBB_tbb_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onetbb_TBB_tbb_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onetbb_TBB_tbb_COMPILE_OPTIONS_C_RELEASE}>")