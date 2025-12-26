########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(pugixml_COMPONENT_NAMES "")
if(DEFINED pugixml_FIND_DEPENDENCY_NAMES)
  list(APPEND pugixml_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES pugixml_FIND_DEPENDENCY_NAMES)
else()
  set(pugixml_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(pugixml_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/pugixe21370abd68b3/p")
set(pugixml_BUILD_MODULES_PATHS_RELEASE )


set(pugixml_INCLUDE_DIRS_RELEASE "${pugixml_PACKAGE_FOLDER_RELEASE}/include")
set(pugixml_RES_DIRS_RELEASE )
set(pugixml_DEFINITIONS_RELEASE )
set(pugixml_SHARED_LINK_FLAGS_RELEASE )
set(pugixml_EXE_LINK_FLAGS_RELEASE )
set(pugixml_OBJECTS_RELEASE )
set(pugixml_COMPILE_DEFINITIONS_RELEASE )
set(pugixml_COMPILE_OPTIONS_C_RELEASE )
set(pugixml_COMPILE_OPTIONS_CXX_RELEASE )
set(pugixml_LIB_DIRS_RELEASE "${pugixml_PACKAGE_FOLDER_RELEASE}/lib")
set(pugixml_BIN_DIRS_RELEASE )
set(pugixml_LIBRARY_TYPE_RELEASE STATIC)
set(pugixml_IS_HOST_WINDOWS_RELEASE 0)
set(pugixml_LIBS_RELEASE pugixml)
set(pugixml_SYSTEM_LIBS_RELEASE m)
set(pugixml_FRAMEWORK_DIRS_RELEASE )
set(pugixml_FRAMEWORKS_RELEASE )
set(pugixml_BUILD_DIRS_RELEASE )
set(pugixml_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(pugixml_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pugixml_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pugixml_COMPILE_OPTIONS_C_RELEASE}>")
set(pugixml_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pugixml_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pugixml_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pugixml_EXE_LINK_FLAGS_RELEASE}>")


set(pugixml_COMPONENTS_RELEASE )