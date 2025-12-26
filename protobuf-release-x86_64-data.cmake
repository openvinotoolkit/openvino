########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND protobuf_COMPONENT_NAMES protobuf::libprotobuf protobuf::libprotobuf-lite protobuf::libprotoc)
list(REMOVE_DUPLICATES protobuf_COMPONENT_NAMES)
if(DEFINED protobuf_FIND_DEPENDENCY_NAMES)
  list(APPEND protobuf_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES protobuf_FIND_DEPENDENCY_NAMES)
else()
  set(protobuf_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(protobuf_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/proto24027a0252bc9/p")
set(protobuf_BUILD_MODULES_PATHS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/lib/cmake/protobuf/protobuf-generate.cmake"
			"${protobuf_PACKAGE_FOLDER_RELEASE}/lib/cmake/protobuf/protobuf-module.cmake"
			"${protobuf_PACKAGE_FOLDER_RELEASE}/lib/cmake/protobuf/protobuf-options.cmake")


set(protobuf_INCLUDE_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/include")
set(protobuf_RES_DIRS_RELEASE )
set(protobuf_DEFINITIONS_RELEASE )
set(protobuf_SHARED_LINK_FLAGS_RELEASE )
set(protobuf_EXE_LINK_FLAGS_RELEASE )
set(protobuf_OBJECTS_RELEASE )
set(protobuf_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_COMPILE_OPTIONS_C_RELEASE )
set(protobuf_COMPILE_OPTIONS_CXX_RELEASE )
set(protobuf_LIB_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/lib")
set(protobuf_BIN_DIRS_RELEASE )
set(protobuf_LIBRARY_TYPE_RELEASE STATIC)
set(protobuf_IS_HOST_WINDOWS_RELEASE 0)
set(protobuf_LIBS_RELEASE protoc protobuf-lite protobuf)
set(protobuf_SYSTEM_LIBS_RELEASE m pthread)
set(protobuf_FRAMEWORK_DIRS_RELEASE )
set(protobuf_FRAMEWORKS_RELEASE )
set(protobuf_BUILD_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/lib/cmake/protobuf")
set(protobuf_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(protobuf_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${protobuf_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${protobuf_COMPILE_OPTIONS_C_RELEASE}>")
set(protobuf_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${protobuf_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${protobuf_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${protobuf_EXE_LINK_FLAGS_RELEASE}>")


set(protobuf_COMPONENTS_RELEASE protobuf::libprotobuf protobuf::libprotobuf-lite protobuf::libprotoc)
########### COMPONENT protobuf::libprotoc VARIABLES ############################################

set(protobuf_protobuf_libprotoc_INCLUDE_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/include")
set(protobuf_protobuf_libprotoc_LIB_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/lib")
set(protobuf_protobuf_libprotoc_BIN_DIRS_RELEASE )
set(protobuf_protobuf_libprotoc_LIBRARY_TYPE_RELEASE STATIC)
set(protobuf_protobuf_libprotoc_IS_HOST_WINDOWS_RELEASE 0)
set(protobuf_protobuf_libprotoc_RES_DIRS_RELEASE )
set(protobuf_protobuf_libprotoc_DEFINITIONS_RELEASE )
set(protobuf_protobuf_libprotoc_OBJECTS_RELEASE )
set(protobuf_protobuf_libprotoc_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_protobuf_libprotoc_COMPILE_OPTIONS_C_RELEASE "")
set(protobuf_protobuf_libprotoc_COMPILE_OPTIONS_CXX_RELEASE "")
set(protobuf_protobuf_libprotoc_LIBS_RELEASE protoc)
set(protobuf_protobuf_libprotoc_SYSTEM_LIBS_RELEASE )
set(protobuf_protobuf_libprotoc_FRAMEWORK_DIRS_RELEASE )
set(protobuf_protobuf_libprotoc_FRAMEWORKS_RELEASE )
set(protobuf_protobuf_libprotoc_DEPENDENCIES_RELEASE protobuf::libprotobuf)
set(protobuf_protobuf_libprotoc_SHARED_LINK_FLAGS_RELEASE )
set(protobuf_protobuf_libprotoc_EXE_LINK_FLAGS_RELEASE )
set(protobuf_protobuf_libprotoc_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(protobuf_protobuf_libprotoc_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${protobuf_protobuf_libprotoc_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${protobuf_protobuf_libprotoc_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${protobuf_protobuf_libprotoc_EXE_LINK_FLAGS_RELEASE}>
)
set(protobuf_protobuf_libprotoc_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${protobuf_protobuf_libprotoc_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${protobuf_protobuf_libprotoc_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT protobuf::libprotobuf-lite VARIABLES ############################################

set(protobuf_protobuf_libprotobuf-lite_INCLUDE_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/include")
set(protobuf_protobuf_libprotobuf-lite_LIB_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/lib")
set(protobuf_protobuf_libprotobuf-lite_BIN_DIRS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_LIBRARY_TYPE_RELEASE STATIC)
set(protobuf_protobuf_libprotobuf-lite_IS_HOST_WINDOWS_RELEASE 0)
set(protobuf_protobuf_libprotobuf-lite_RES_DIRS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_DEFINITIONS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_OBJECTS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_COMPILE_OPTIONS_C_RELEASE "")
set(protobuf_protobuf_libprotobuf-lite_COMPILE_OPTIONS_CXX_RELEASE "")
set(protobuf_protobuf_libprotobuf-lite_LIBS_RELEASE protobuf-lite)
set(protobuf_protobuf_libprotobuf-lite_SYSTEM_LIBS_RELEASE m pthread)
set(protobuf_protobuf_libprotobuf-lite_FRAMEWORK_DIRS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_FRAMEWORKS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_DEPENDENCIES_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_SHARED_LINK_FLAGS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_EXE_LINK_FLAGS_RELEASE )
set(protobuf_protobuf_libprotobuf-lite_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(protobuf_protobuf_libprotobuf-lite_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${protobuf_protobuf_libprotobuf-lite_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${protobuf_protobuf_libprotobuf-lite_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${protobuf_protobuf_libprotobuf-lite_EXE_LINK_FLAGS_RELEASE}>
)
set(protobuf_protobuf_libprotobuf-lite_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${protobuf_protobuf_libprotobuf-lite_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${protobuf_protobuf_libprotobuf-lite_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT protobuf::libprotobuf VARIABLES ############################################

set(protobuf_protobuf_libprotobuf_INCLUDE_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/include")
set(protobuf_protobuf_libprotobuf_LIB_DIRS_RELEASE "${protobuf_PACKAGE_FOLDER_RELEASE}/lib")
set(protobuf_protobuf_libprotobuf_BIN_DIRS_RELEASE )
set(protobuf_protobuf_libprotobuf_LIBRARY_TYPE_RELEASE STATIC)
set(protobuf_protobuf_libprotobuf_IS_HOST_WINDOWS_RELEASE 0)
set(protobuf_protobuf_libprotobuf_RES_DIRS_RELEASE )
set(protobuf_protobuf_libprotobuf_DEFINITIONS_RELEASE )
set(protobuf_protobuf_libprotobuf_OBJECTS_RELEASE )
set(protobuf_protobuf_libprotobuf_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_protobuf_libprotobuf_COMPILE_OPTIONS_C_RELEASE "")
set(protobuf_protobuf_libprotobuf_COMPILE_OPTIONS_CXX_RELEASE "")
set(protobuf_protobuf_libprotobuf_LIBS_RELEASE protobuf)
set(protobuf_protobuf_libprotobuf_SYSTEM_LIBS_RELEASE m pthread)
set(protobuf_protobuf_libprotobuf_FRAMEWORK_DIRS_RELEASE )
set(protobuf_protobuf_libprotobuf_FRAMEWORKS_RELEASE )
set(protobuf_protobuf_libprotobuf_DEPENDENCIES_RELEASE )
set(protobuf_protobuf_libprotobuf_SHARED_LINK_FLAGS_RELEASE )
set(protobuf_protobuf_libprotobuf_EXE_LINK_FLAGS_RELEASE )
set(protobuf_protobuf_libprotobuf_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(protobuf_protobuf_libprotobuf_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${protobuf_protobuf_libprotobuf_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${protobuf_protobuf_libprotobuf_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${protobuf_protobuf_libprotobuf_EXE_LINK_FLAGS_RELEASE}>
)
set(protobuf_protobuf_libprotobuf_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${protobuf_protobuf_libprotobuf_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${protobuf_protobuf_libprotobuf_COMPILE_OPTIONS_C_RELEASE}>")