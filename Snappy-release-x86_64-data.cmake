########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND snappy_COMPONENT_NAMES Snappy::snappy)
list(REMOVE_DUPLICATES snappy_COMPONENT_NAMES)
if(DEFINED snappy_FIND_DEPENDENCY_NAMES)
  list(APPEND snappy_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES snappy_FIND_DEPENDENCY_NAMES)
else()
  set(snappy_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(snappy_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/b/snappb44f66fc2baca/p")
set(snappy_BUILD_MODULES_PATHS_RELEASE )


set(snappy_INCLUDE_DIRS_RELEASE "${snappy_PACKAGE_FOLDER_RELEASE}/include")
set(snappy_RES_DIRS_RELEASE )
set(snappy_DEFINITIONS_RELEASE )
set(snappy_SHARED_LINK_FLAGS_RELEASE )
set(snappy_EXE_LINK_FLAGS_RELEASE )
set(snappy_OBJECTS_RELEASE )
set(snappy_COMPILE_DEFINITIONS_RELEASE )
set(snappy_COMPILE_OPTIONS_C_RELEASE )
set(snappy_COMPILE_OPTIONS_CXX_RELEASE )
set(snappy_LIB_DIRS_RELEASE "${snappy_PACKAGE_FOLDER_RELEASE}/lib")
set(snappy_BIN_DIRS_RELEASE )
set(snappy_LIBRARY_TYPE_RELEASE STATIC)
set(snappy_IS_HOST_WINDOWS_RELEASE 0)
set(snappy_LIBS_RELEASE snappy)
set(snappy_SYSTEM_LIBS_RELEASE m stdc++)
set(snappy_FRAMEWORK_DIRS_RELEASE )
set(snappy_FRAMEWORKS_RELEASE )
set(snappy_BUILD_DIRS_RELEASE )
set(snappy_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(snappy_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${snappy_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${snappy_COMPILE_OPTIONS_C_RELEASE}>")
set(snappy_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${snappy_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${snappy_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${snappy_EXE_LINK_FLAGS_RELEASE}>")


set(snappy_COMPONENTS_RELEASE Snappy::snappy)
########### COMPONENT Snappy::snappy VARIABLES ############################################

set(snappy_Snappy_snappy_INCLUDE_DIRS_RELEASE "${snappy_PACKAGE_FOLDER_RELEASE}/include")
set(snappy_Snappy_snappy_LIB_DIRS_RELEASE "${snappy_PACKAGE_FOLDER_RELEASE}/lib")
set(snappy_Snappy_snappy_BIN_DIRS_RELEASE )
set(snappy_Snappy_snappy_LIBRARY_TYPE_RELEASE STATIC)
set(snappy_Snappy_snappy_IS_HOST_WINDOWS_RELEASE 0)
set(snappy_Snappy_snappy_RES_DIRS_RELEASE )
set(snappy_Snappy_snappy_DEFINITIONS_RELEASE )
set(snappy_Snappy_snappy_OBJECTS_RELEASE )
set(snappy_Snappy_snappy_COMPILE_DEFINITIONS_RELEASE )
set(snappy_Snappy_snappy_COMPILE_OPTIONS_C_RELEASE "")
set(snappy_Snappy_snappy_COMPILE_OPTIONS_CXX_RELEASE "")
set(snappy_Snappy_snappy_LIBS_RELEASE snappy)
set(snappy_Snappy_snappy_SYSTEM_LIBS_RELEASE m stdc++)
set(snappy_Snappy_snappy_FRAMEWORK_DIRS_RELEASE )
set(snappy_Snappy_snappy_FRAMEWORKS_RELEASE )
set(snappy_Snappy_snappy_DEPENDENCIES_RELEASE )
set(snappy_Snappy_snappy_SHARED_LINK_FLAGS_RELEASE )
set(snappy_Snappy_snappy_EXE_LINK_FLAGS_RELEASE )
set(snappy_Snappy_snappy_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(snappy_Snappy_snappy_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${snappy_Snappy_snappy_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${snappy_Snappy_snappy_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${snappy_Snappy_snappy_EXE_LINK_FLAGS_RELEASE}>
)
set(snappy_Snappy_snappy_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${snappy_Snappy_snappy_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${snappy_Snappy_snappy_COMPILE_OPTIONS_C_RELEASE}>")