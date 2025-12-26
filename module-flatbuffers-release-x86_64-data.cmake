########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND flatbuffers_COMPONENT_NAMES flatbuffers::libflatbuffers)
list(REMOVE_DUPLICATES flatbuffers_COMPONENT_NAMES)
if(DEFINED flatbuffers_FIND_DEPENDENCY_NAMES)
  list(APPEND flatbuffers_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES flatbuffers_FIND_DEPENDENCY_NAMES)
else()
  set(flatbuffers_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(flatbuffers_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/flatbf8008cb9f3e3a/p")
set(flatbuffers_BUILD_MODULES_PATHS_RELEASE "${flatbuffers_PACKAGE_FOLDER_RELEASE}/lib/cmake/FlatcTargets.cmake"
			"${flatbuffers_PACKAGE_FOLDER_RELEASE}/lib/cmake/BuildFlatBuffers.cmake")


set(flatbuffers_INCLUDE_DIRS_RELEASE "${flatbuffers_PACKAGE_FOLDER_RELEASE}/include")
set(flatbuffers_RES_DIRS_RELEASE )
set(flatbuffers_DEFINITIONS_RELEASE )
set(flatbuffers_SHARED_LINK_FLAGS_RELEASE )
set(flatbuffers_EXE_LINK_FLAGS_RELEASE )
set(flatbuffers_OBJECTS_RELEASE )
set(flatbuffers_COMPILE_DEFINITIONS_RELEASE )
set(flatbuffers_COMPILE_OPTIONS_C_RELEASE )
set(flatbuffers_COMPILE_OPTIONS_CXX_RELEASE )
set(flatbuffers_LIB_DIRS_RELEASE "${flatbuffers_PACKAGE_FOLDER_RELEASE}/lib")
set(flatbuffers_BIN_DIRS_RELEASE )
set(flatbuffers_LIBRARY_TYPE_RELEASE UNKNOWN)
set(flatbuffers_IS_HOST_WINDOWS_RELEASE 0)
set(flatbuffers_LIBS_RELEASE )
set(flatbuffers_SYSTEM_LIBS_RELEASE )
set(flatbuffers_FRAMEWORK_DIRS_RELEASE )
set(flatbuffers_FRAMEWORKS_RELEASE )
set(flatbuffers_BUILD_DIRS_RELEASE )
set(flatbuffers_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(flatbuffers_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${flatbuffers_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${flatbuffers_COMPILE_OPTIONS_C_RELEASE}>")
set(flatbuffers_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${flatbuffers_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${flatbuffers_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${flatbuffers_EXE_LINK_FLAGS_RELEASE}>")


set(flatbuffers_COMPONENTS_RELEASE flatbuffers::libflatbuffers)
########### COMPONENT flatbuffers::libflatbuffers VARIABLES ############################################

set(flatbuffers_flatbuffers_libflatbuffers_INCLUDE_DIRS_RELEASE "${flatbuffers_PACKAGE_FOLDER_RELEASE}/include")
set(flatbuffers_flatbuffers_libflatbuffers_LIB_DIRS_RELEASE "${flatbuffers_PACKAGE_FOLDER_RELEASE}/lib")
set(flatbuffers_flatbuffers_libflatbuffers_BIN_DIRS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_LIBRARY_TYPE_RELEASE UNKNOWN)
set(flatbuffers_flatbuffers_libflatbuffers_IS_HOST_WINDOWS_RELEASE 0)
set(flatbuffers_flatbuffers_libflatbuffers_RES_DIRS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_DEFINITIONS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_OBJECTS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_COMPILE_DEFINITIONS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_COMPILE_OPTIONS_C_RELEASE "")
set(flatbuffers_flatbuffers_libflatbuffers_COMPILE_OPTIONS_CXX_RELEASE "")
set(flatbuffers_flatbuffers_libflatbuffers_LIBS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_SYSTEM_LIBS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_FRAMEWORK_DIRS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_FRAMEWORKS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_DEPENDENCIES_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_SHARED_LINK_FLAGS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_EXE_LINK_FLAGS_RELEASE )
set(flatbuffers_flatbuffers_libflatbuffers_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(flatbuffers_flatbuffers_libflatbuffers_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${flatbuffers_flatbuffers_libflatbuffers_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${flatbuffers_flatbuffers_libflatbuffers_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${flatbuffers_flatbuffers_libflatbuffers_EXE_LINK_FLAGS_RELEASE}>
)
set(flatbuffers_flatbuffers_libflatbuffers_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${flatbuffers_flatbuffers_libflatbuffers_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${flatbuffers_flatbuffers_libflatbuffers_COMPILE_OPTIONS_C_RELEASE}>")