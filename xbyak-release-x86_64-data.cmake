########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(xbyak_COMPONENT_NAMES "")
if(DEFINED xbyak_FIND_DEPENDENCY_NAMES)
  list(APPEND xbyak_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES xbyak_FIND_DEPENDENCY_NAMES)
else()
  set(xbyak_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(xbyak_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/xbyak509ed9bc3a8cf/p")
set(xbyak_BUILD_MODULES_PATHS_RELEASE )


set(xbyak_INCLUDE_DIRS_RELEASE "${xbyak_PACKAGE_FOLDER_RELEASE}/include")
set(xbyak_RES_DIRS_RELEASE )
set(xbyak_DEFINITIONS_RELEASE )
set(xbyak_SHARED_LINK_FLAGS_RELEASE )
set(xbyak_EXE_LINK_FLAGS_RELEASE )
set(xbyak_OBJECTS_RELEASE )
set(xbyak_COMPILE_DEFINITIONS_RELEASE )
set(xbyak_COMPILE_OPTIONS_C_RELEASE )
set(xbyak_COMPILE_OPTIONS_CXX_RELEASE )
set(xbyak_LIB_DIRS_RELEASE )
set(xbyak_BIN_DIRS_RELEASE )
set(xbyak_LIBRARY_TYPE_RELEASE UNKNOWN)
set(xbyak_IS_HOST_WINDOWS_RELEASE 0)
set(xbyak_LIBS_RELEASE )
set(xbyak_SYSTEM_LIBS_RELEASE )
set(xbyak_FRAMEWORK_DIRS_RELEASE )
set(xbyak_FRAMEWORKS_RELEASE )
set(xbyak_BUILD_DIRS_RELEASE )
set(xbyak_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(xbyak_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${xbyak_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${xbyak_COMPILE_OPTIONS_C_RELEASE}>")
set(xbyak_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${xbyak_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${xbyak_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${xbyak_EXE_LINK_FLAGS_RELEASE}>")


set(xbyak_COMPONENTS_RELEASE )