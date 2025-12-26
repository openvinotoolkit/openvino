########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(opencl-headers_COMPONENT_NAMES "")
if(DEFINED opencl-headers_FIND_DEPENDENCY_NAMES)
  list(APPEND opencl-headers_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES opencl-headers_FIND_DEPENDENCY_NAMES)
else()
  set(opencl-headers_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(opencl-headers_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/openc486b02081d5a8/p")
set(opencl-headers_BUILD_MODULES_PATHS_RELEASE )


set(opencl-headers_INCLUDE_DIRS_RELEASE "${opencl-headers_PACKAGE_FOLDER_RELEASE}/include")
set(opencl-headers_RES_DIRS_RELEASE )
set(opencl-headers_DEFINITIONS_RELEASE )
set(opencl-headers_SHARED_LINK_FLAGS_RELEASE )
set(opencl-headers_EXE_LINK_FLAGS_RELEASE )
set(opencl-headers_OBJECTS_RELEASE )
set(opencl-headers_COMPILE_DEFINITIONS_RELEASE )
set(opencl-headers_COMPILE_OPTIONS_C_RELEASE )
set(opencl-headers_COMPILE_OPTIONS_CXX_RELEASE )
set(opencl-headers_LIB_DIRS_RELEASE )
set(opencl-headers_BIN_DIRS_RELEASE )
set(opencl-headers_LIBRARY_TYPE_RELEASE UNKNOWN)
set(opencl-headers_IS_HOST_WINDOWS_RELEASE 0)
set(opencl-headers_LIBS_RELEASE )
set(opencl-headers_SYSTEM_LIBS_RELEASE )
set(opencl-headers_FRAMEWORK_DIRS_RELEASE )
set(opencl-headers_FRAMEWORKS_RELEASE )
set(opencl-headers_BUILD_DIRS_RELEASE )
set(opencl-headers_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(opencl-headers_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencl-headers_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencl-headers_COMPILE_OPTIONS_C_RELEASE}>")
set(opencl-headers_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencl-headers_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencl-headers_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencl-headers_EXE_LINK_FLAGS_RELEASE}>")


set(opencl-headers_COMPONENTS_RELEASE )