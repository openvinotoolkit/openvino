########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(opencl-clhpp-headers_COMPONENT_NAMES "")
if(DEFINED opencl-clhpp-headers_FIND_DEPENDENCY_NAMES)
  list(APPEND opencl-clhpp-headers_FIND_DEPENDENCY_NAMES OpenCLHeaders)
  list(REMOVE_DUPLICATES opencl-clhpp-headers_FIND_DEPENDENCY_NAMES)
else()
  set(opencl-clhpp-headers_FIND_DEPENDENCY_NAMES OpenCLHeaders)
endif()
set(OpenCLHeaders_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(opencl-clhpp-headers_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/openca550c77954569/p")
set(opencl-clhpp-headers_BUILD_MODULES_PATHS_RELEASE )


set(opencl-clhpp-headers_INCLUDE_DIRS_RELEASE "${opencl-clhpp-headers_PACKAGE_FOLDER_RELEASE}/include")
set(opencl-clhpp-headers_RES_DIRS_RELEASE )
set(opencl-clhpp-headers_DEFINITIONS_RELEASE )
set(opencl-clhpp-headers_SHARED_LINK_FLAGS_RELEASE )
set(opencl-clhpp-headers_EXE_LINK_FLAGS_RELEASE )
set(opencl-clhpp-headers_OBJECTS_RELEASE )
set(opencl-clhpp-headers_COMPILE_DEFINITIONS_RELEASE )
set(opencl-clhpp-headers_COMPILE_OPTIONS_C_RELEASE )
set(opencl-clhpp-headers_COMPILE_OPTIONS_CXX_RELEASE )
set(opencl-clhpp-headers_LIB_DIRS_RELEASE )
set(opencl-clhpp-headers_BIN_DIRS_RELEASE )
set(opencl-clhpp-headers_LIBRARY_TYPE_RELEASE UNKNOWN)
set(opencl-clhpp-headers_IS_HOST_WINDOWS_RELEASE 0)
set(opencl-clhpp-headers_LIBS_RELEASE )
set(opencl-clhpp-headers_SYSTEM_LIBS_RELEASE )
set(opencl-clhpp-headers_FRAMEWORK_DIRS_RELEASE )
set(opencl-clhpp-headers_FRAMEWORKS_RELEASE )
set(opencl-clhpp-headers_BUILD_DIRS_RELEASE )
set(opencl-clhpp-headers_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(opencl-clhpp-headers_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencl-clhpp-headers_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencl-clhpp-headers_COMPILE_OPTIONS_C_RELEASE}>")
set(opencl-clhpp-headers_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencl-clhpp-headers_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencl-clhpp-headers_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencl-clhpp-headers_EXE_LINK_FLAGS_RELEASE}>")


set(opencl-clhpp-headers_COMPONENTS_RELEASE )