########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(rapidjson_COMPONENT_NAMES "")
if(DEFINED rapidjson_FIND_DEPENDENCY_NAMES)
  list(APPEND rapidjson_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES rapidjson_FIND_DEPENDENCY_NAMES)
else()
  set(rapidjson_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(rapidjson_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/rapid4cabb31a09329/p")
set(rapidjson_BUILD_MODULES_PATHS_RELEASE )


set(rapidjson_INCLUDE_DIRS_RELEASE "${rapidjson_PACKAGE_FOLDER_RELEASE}/include")
set(rapidjson_RES_DIRS_RELEASE )
set(rapidjson_DEFINITIONS_RELEASE )
set(rapidjson_SHARED_LINK_FLAGS_RELEASE )
set(rapidjson_EXE_LINK_FLAGS_RELEASE )
set(rapidjson_OBJECTS_RELEASE )
set(rapidjson_COMPILE_DEFINITIONS_RELEASE )
set(rapidjson_COMPILE_OPTIONS_C_RELEASE )
set(rapidjson_COMPILE_OPTIONS_CXX_RELEASE )
set(rapidjson_LIB_DIRS_RELEASE "${rapidjson_PACKAGE_FOLDER_RELEASE}/lib")
set(rapidjson_BIN_DIRS_RELEASE )
set(rapidjson_LIBRARY_TYPE_RELEASE UNKNOWN)
set(rapidjson_IS_HOST_WINDOWS_RELEASE 0)
set(rapidjson_LIBS_RELEASE )
set(rapidjson_SYSTEM_LIBS_RELEASE )
set(rapidjson_FRAMEWORK_DIRS_RELEASE )
set(rapidjson_FRAMEWORKS_RELEASE )
set(rapidjson_BUILD_DIRS_RELEASE )
set(rapidjson_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(rapidjson_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${rapidjson_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${rapidjson_COMPILE_OPTIONS_C_RELEASE}>")
set(rapidjson_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${rapidjson_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${rapidjson_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${rapidjson_EXE_LINK_FLAGS_RELEASE}>")


set(rapidjson_COMPONENTS_RELEASE )