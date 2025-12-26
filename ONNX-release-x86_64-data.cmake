########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND onnx_COMPONENT_NAMES onnx_proto onnx)
list(REMOVE_DUPLICATES onnx_COMPONENT_NAMES)
if(DEFINED onnx_FIND_DEPENDENCY_NAMES)
  list(APPEND onnx_FIND_DEPENDENCY_NAMES protobuf)
  list(REMOVE_DUPLICATES onnx_FIND_DEPENDENCY_NAMES)
else()
  set(onnx_FIND_DEPENDENCY_NAMES protobuf)
endif()
set(protobuf_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(onnx_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/onnxbecb5e921cdaa/p")
set(onnx_BUILD_MODULES_PATHS_RELEASE )


set(onnx_INCLUDE_DIRS_RELEASE "${onnx_PACKAGE_FOLDER_RELEASE}/include")
set(onnx_RES_DIRS_RELEASE )
set(onnx_DEFINITIONS_RELEASE "-DONNX_NAMESPACE=onnx"
			"-DONNX_ML=1"
			"-D__STDC_FORMAT_MACROS")
set(onnx_SHARED_LINK_FLAGS_RELEASE )
set(onnx_EXE_LINK_FLAGS_RELEASE )
set(onnx_OBJECTS_RELEASE )
set(onnx_COMPILE_DEFINITIONS_RELEASE "ONNX_NAMESPACE=onnx"
			"ONNX_ML=1"
			"__STDC_FORMAT_MACROS")
set(onnx_COMPILE_OPTIONS_C_RELEASE )
set(onnx_COMPILE_OPTIONS_CXX_RELEASE )
set(onnx_LIB_DIRS_RELEASE "${onnx_PACKAGE_FOLDER_RELEASE}/lib")
set(onnx_BIN_DIRS_RELEASE )
set(onnx_LIBRARY_TYPE_RELEASE STATIC)
set(onnx_IS_HOST_WINDOWS_RELEASE 0)
set(onnx_LIBS_RELEASE onnx onnx_proto)
set(onnx_SYSTEM_LIBS_RELEASE )
set(onnx_FRAMEWORK_DIRS_RELEASE )
set(onnx_FRAMEWORKS_RELEASE )
set(onnx_BUILD_DIRS_RELEASE )
set(onnx_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(onnx_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onnx_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onnx_COMPILE_OPTIONS_C_RELEASE}>")
set(onnx_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onnx_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onnx_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onnx_EXE_LINK_FLAGS_RELEASE}>")


set(onnx_COMPONENTS_RELEASE onnx_proto onnx)
########### COMPONENT onnx VARIABLES ############################################

set(onnx_onnx_INCLUDE_DIRS_RELEASE "${onnx_PACKAGE_FOLDER_RELEASE}/include")
set(onnx_onnx_LIB_DIRS_RELEASE "${onnx_PACKAGE_FOLDER_RELEASE}/lib")
set(onnx_onnx_BIN_DIRS_RELEASE )
set(onnx_onnx_LIBRARY_TYPE_RELEASE STATIC)
set(onnx_onnx_IS_HOST_WINDOWS_RELEASE 0)
set(onnx_onnx_RES_DIRS_RELEASE )
set(onnx_onnx_DEFINITIONS_RELEASE "-DONNX_NAMESPACE=onnx"
			"-DONNX_ML=1"
			"-D__STDC_FORMAT_MACROS")
set(onnx_onnx_OBJECTS_RELEASE )
set(onnx_onnx_COMPILE_DEFINITIONS_RELEASE "ONNX_NAMESPACE=onnx"
			"ONNX_ML=1"
			"__STDC_FORMAT_MACROS")
set(onnx_onnx_COMPILE_OPTIONS_C_RELEASE "")
set(onnx_onnx_COMPILE_OPTIONS_CXX_RELEASE "")
set(onnx_onnx_LIBS_RELEASE onnx)
set(onnx_onnx_SYSTEM_LIBS_RELEASE )
set(onnx_onnx_FRAMEWORK_DIRS_RELEASE )
set(onnx_onnx_FRAMEWORKS_RELEASE )
set(onnx_onnx_DEPENDENCIES_RELEASE onnx_proto)
set(onnx_onnx_SHARED_LINK_FLAGS_RELEASE )
set(onnx_onnx_EXE_LINK_FLAGS_RELEASE )
set(onnx_onnx_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(onnx_onnx_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onnx_onnx_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onnx_onnx_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onnx_onnx_EXE_LINK_FLAGS_RELEASE}>
)
set(onnx_onnx_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onnx_onnx_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onnx_onnx_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT onnx_proto VARIABLES ############################################

set(onnx_onnx_proto_INCLUDE_DIRS_RELEASE "${onnx_PACKAGE_FOLDER_RELEASE}/include")
set(onnx_onnx_proto_LIB_DIRS_RELEASE "${onnx_PACKAGE_FOLDER_RELEASE}/lib")
set(onnx_onnx_proto_BIN_DIRS_RELEASE )
set(onnx_onnx_proto_LIBRARY_TYPE_RELEASE STATIC)
set(onnx_onnx_proto_IS_HOST_WINDOWS_RELEASE 0)
set(onnx_onnx_proto_RES_DIRS_RELEASE )
set(onnx_onnx_proto_DEFINITIONS_RELEASE "-DONNX_NAMESPACE=onnx"
			"-DONNX_ML=1")
set(onnx_onnx_proto_OBJECTS_RELEASE )
set(onnx_onnx_proto_COMPILE_DEFINITIONS_RELEASE "ONNX_NAMESPACE=onnx"
			"ONNX_ML=1")
set(onnx_onnx_proto_COMPILE_OPTIONS_C_RELEASE "")
set(onnx_onnx_proto_COMPILE_OPTIONS_CXX_RELEASE "")
set(onnx_onnx_proto_LIBS_RELEASE onnx_proto)
set(onnx_onnx_proto_SYSTEM_LIBS_RELEASE )
set(onnx_onnx_proto_FRAMEWORK_DIRS_RELEASE )
set(onnx_onnx_proto_FRAMEWORKS_RELEASE )
set(onnx_onnx_proto_DEPENDENCIES_RELEASE protobuf::libprotobuf)
set(onnx_onnx_proto_SHARED_LINK_FLAGS_RELEASE )
set(onnx_onnx_proto_EXE_LINK_FLAGS_RELEASE )
set(onnx_onnx_proto_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(onnx_onnx_proto_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${onnx_onnx_proto_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${onnx_onnx_proto_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${onnx_onnx_proto_EXE_LINK_FLAGS_RELEASE}>
)
set(onnx_onnx_proto_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${onnx_onnx_proto_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${onnx_onnx_proto_COMPILE_OPTIONS_C_RELEASE}>")