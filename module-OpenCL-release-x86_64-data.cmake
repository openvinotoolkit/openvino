########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(opencl-icd-loader_COMPONENT_NAMES "")
if(DEFINED opencl-icd-loader_FIND_DEPENDENCY_NAMES)
  list(APPEND opencl-icd-loader_FIND_DEPENDENCY_NAMES OpenCLHeadersCpp OpenCLHeaders)
  list(REMOVE_DUPLICATES opencl-icd-loader_FIND_DEPENDENCY_NAMES)
else()
  set(opencl-icd-loader_FIND_DEPENDENCY_NAMES OpenCLHeadersCpp OpenCLHeaders)
endif()
set(OpenCLHeadersCpp_FIND_MODE "NO_MODULE")
set(OpenCLHeaders_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(opencl-icd-loader_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/openc88de3f809feb7/p")
set(opencl-icd-loader_BUILD_MODULES_PATHS_RELEASE )


set(opencl-icd-loader_INCLUDE_DIRS_RELEASE )
set(opencl-icd-loader_RES_DIRS_RELEASE )
set(opencl-icd-loader_DEFINITIONS_RELEASE )
set(opencl-icd-loader_SHARED_LINK_FLAGS_RELEASE )
set(opencl-icd-loader_EXE_LINK_FLAGS_RELEASE )
set(opencl-icd-loader_OBJECTS_RELEASE )
set(opencl-icd-loader_COMPILE_DEFINITIONS_RELEASE )
set(opencl-icd-loader_COMPILE_OPTIONS_C_RELEASE )
set(opencl-icd-loader_COMPILE_OPTIONS_CXX_RELEASE )
set(opencl-icd-loader_LIB_DIRS_RELEASE "${opencl-icd-loader_PACKAGE_FOLDER_RELEASE}/lib")
set(opencl-icd-loader_BIN_DIRS_RELEASE )
set(opencl-icd-loader_LIBRARY_TYPE_RELEASE STATIC)
set(opencl-icd-loader_IS_HOST_WINDOWS_RELEASE 0)
set(opencl-icd-loader_LIBS_RELEASE OpenCL)
set(opencl-icd-loader_SYSTEM_LIBS_RELEASE dl pthread)
set(opencl-icd-loader_FRAMEWORK_DIRS_RELEASE )
set(opencl-icd-loader_FRAMEWORKS_RELEASE )
set(opencl-icd-loader_BUILD_DIRS_RELEASE )
set(opencl-icd-loader_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(opencl-icd-loader_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencl-icd-loader_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencl-icd-loader_COMPILE_OPTIONS_C_RELEASE}>")
set(opencl-icd-loader_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencl-icd-loader_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencl-icd-loader_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencl-icd-loader_EXE_LINK_FLAGS_RELEASE}>")


set(opencl-icd-loader_COMPONENTS_RELEASE )