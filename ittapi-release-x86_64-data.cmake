########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(ittapi_COMPONENT_NAMES "")
if(DEFINED ittapi_FIND_DEPENDENCY_NAMES)
  list(APPEND ittapi_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES ittapi_FIND_DEPENDENCY_NAMES)
else()
  set(ittapi_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(ittapi_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/ittapd75c990bd22eb/p")
set(ittapi_BUILD_MODULES_PATHS_RELEASE )


set(ittapi_INCLUDE_DIRS_RELEASE "${ittapi_PACKAGE_FOLDER_RELEASE}/include")
set(ittapi_RES_DIRS_RELEASE )
set(ittapi_DEFINITIONS_RELEASE )
set(ittapi_SHARED_LINK_FLAGS_RELEASE )
set(ittapi_EXE_LINK_FLAGS_RELEASE )
set(ittapi_OBJECTS_RELEASE )
set(ittapi_COMPILE_DEFINITIONS_RELEASE )
set(ittapi_COMPILE_OPTIONS_C_RELEASE )
set(ittapi_COMPILE_OPTIONS_CXX_RELEASE )
set(ittapi_LIB_DIRS_RELEASE "${ittapi_PACKAGE_FOLDER_RELEASE}/lib")
set(ittapi_BIN_DIRS_RELEASE )
set(ittapi_LIBRARY_TYPE_RELEASE UNKNOWN)
set(ittapi_IS_HOST_WINDOWS_RELEASE 0)
set(ittapi_LIBS_RELEASE ittnotify)
set(ittapi_SYSTEM_LIBS_RELEASE dl)
set(ittapi_FRAMEWORK_DIRS_RELEASE )
set(ittapi_FRAMEWORKS_RELEASE )
set(ittapi_BUILD_DIRS_RELEASE )
set(ittapi_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(ittapi_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${ittapi_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${ittapi_COMPILE_OPTIONS_C_RELEASE}>")
set(ittapi_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${ittapi_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${ittapi_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${ittapi_EXE_LINK_FLAGS_RELEASE}>")


set(ittapi_COMPONENTS_RELEASE )