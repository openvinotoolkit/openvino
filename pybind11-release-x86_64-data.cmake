########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND pybind11_COMPONENT_NAMES pybind11::headers pybind11::pybind11 pybind11::embed pybind11::module pybind11::python_link_helper pybind11::windows_extras pybind11::lto pybind11::thin_lto pybind11::opt_size pybind11::python2_no_register)
list(REMOVE_DUPLICATES pybind11_COMPONENT_NAMES)
if(DEFINED pybind11_FIND_DEPENDENCY_NAMES)
  list(APPEND pybind11_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES pybind11_FIND_DEPENDENCY_NAMES)
else()
  set(pybind11_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(pybind11_PACKAGE_FOLDER_RELEASE "/home/vyomesh/.conan2/p/pybine0913baa04aa6/p")
set(pybind11_BUILD_MODULES_PATHS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib/cmake/pybind11/pybind11Common.cmake")


set(pybind11_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_RES_DIRS_RELEASE )
set(pybind11_DEFINITIONS_RELEASE )
set(pybind11_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_EXE_LINK_FLAGS_RELEASE )
set(pybind11_OBJECTS_RELEASE )
set(pybind11_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_COMPILE_OPTIONS_C_RELEASE )
set(pybind11_COMPILE_OPTIONS_CXX_RELEASE )
set(pybind11_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_BIN_DIRS_RELEASE )
set(pybind11_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_LIBS_RELEASE )
set(pybind11_SYSTEM_LIBS_RELEASE )
set(pybind11_FRAMEWORK_DIRS_RELEASE )
set(pybind11_FRAMEWORKS_RELEASE )
set(pybind11_BUILD_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib/cmake/pybind11")
set(pybind11_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(pybind11_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_COMPILE_OPTIONS_C_RELEASE}>")
set(pybind11_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_EXE_LINK_FLAGS_RELEASE}>")


set(pybind11_COMPONENTS_RELEASE pybind11::headers pybind11::pybind11 pybind11::embed pybind11::module pybind11::python_link_helper pybind11::windows_extras pybind11::lto pybind11::thin_lto pybind11::opt_size pybind11::python2_no_register)
########### COMPONENT pybind11::python2_no_register VARIABLES ############################################

set(pybind11_pybind11_python2_no_register_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_python2_no_register_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_python2_no_register_BIN_DIRS_RELEASE )
set(pybind11_pybind11_python2_no_register_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_python2_no_register_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_python2_no_register_RES_DIRS_RELEASE )
set(pybind11_pybind11_python2_no_register_DEFINITIONS_RELEASE )
set(pybind11_pybind11_python2_no_register_OBJECTS_RELEASE )
set(pybind11_pybind11_python2_no_register_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_python2_no_register_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_python2_no_register_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_python2_no_register_LIBS_RELEASE )
set(pybind11_pybind11_python2_no_register_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_python2_no_register_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_python2_no_register_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_python2_no_register_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_python2_no_register_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_python2_no_register_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_python2_no_register_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_python2_no_register_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_python2_no_register_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_python2_no_register_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_python2_no_register_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_python2_no_register_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_python2_no_register_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_python2_no_register_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::opt_size VARIABLES ############################################

set(pybind11_pybind11_opt_size_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_opt_size_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_opt_size_BIN_DIRS_RELEASE )
set(pybind11_pybind11_opt_size_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_opt_size_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_opt_size_RES_DIRS_RELEASE )
set(pybind11_pybind11_opt_size_DEFINITIONS_RELEASE )
set(pybind11_pybind11_opt_size_OBJECTS_RELEASE )
set(pybind11_pybind11_opt_size_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_opt_size_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_opt_size_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_opt_size_LIBS_RELEASE )
set(pybind11_pybind11_opt_size_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_opt_size_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_opt_size_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_opt_size_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_opt_size_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_opt_size_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_opt_size_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_opt_size_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_opt_size_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_opt_size_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_opt_size_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_opt_size_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_opt_size_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_opt_size_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::thin_lto VARIABLES ############################################

set(pybind11_pybind11_thin_lto_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_thin_lto_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_thin_lto_BIN_DIRS_RELEASE )
set(pybind11_pybind11_thin_lto_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_thin_lto_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_thin_lto_RES_DIRS_RELEASE )
set(pybind11_pybind11_thin_lto_DEFINITIONS_RELEASE )
set(pybind11_pybind11_thin_lto_OBJECTS_RELEASE )
set(pybind11_pybind11_thin_lto_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_thin_lto_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_thin_lto_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_thin_lto_LIBS_RELEASE )
set(pybind11_pybind11_thin_lto_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_thin_lto_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_thin_lto_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_thin_lto_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_thin_lto_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_thin_lto_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_thin_lto_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_thin_lto_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_thin_lto_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_thin_lto_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_thin_lto_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_thin_lto_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_thin_lto_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_thin_lto_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::lto VARIABLES ############################################

set(pybind11_pybind11_lto_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_lto_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_lto_BIN_DIRS_RELEASE )
set(pybind11_pybind11_lto_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_lto_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_lto_RES_DIRS_RELEASE )
set(pybind11_pybind11_lto_DEFINITIONS_RELEASE )
set(pybind11_pybind11_lto_OBJECTS_RELEASE )
set(pybind11_pybind11_lto_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_lto_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_lto_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_lto_LIBS_RELEASE )
set(pybind11_pybind11_lto_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_lto_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_lto_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_lto_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_lto_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_lto_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_lto_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_lto_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_lto_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_lto_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_lto_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_lto_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_lto_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_lto_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::windows_extras VARIABLES ############################################

set(pybind11_pybind11_windows_extras_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_windows_extras_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_windows_extras_BIN_DIRS_RELEASE )
set(pybind11_pybind11_windows_extras_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_windows_extras_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_windows_extras_RES_DIRS_RELEASE )
set(pybind11_pybind11_windows_extras_DEFINITIONS_RELEASE )
set(pybind11_pybind11_windows_extras_OBJECTS_RELEASE )
set(pybind11_pybind11_windows_extras_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_windows_extras_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_windows_extras_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_windows_extras_LIBS_RELEASE )
set(pybind11_pybind11_windows_extras_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_windows_extras_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_windows_extras_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_windows_extras_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_windows_extras_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_windows_extras_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_windows_extras_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_windows_extras_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_windows_extras_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_windows_extras_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_windows_extras_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_windows_extras_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_windows_extras_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_windows_extras_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::python_link_helper VARIABLES ############################################

set(pybind11_pybind11_python_link_helper_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_python_link_helper_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_python_link_helper_BIN_DIRS_RELEASE )
set(pybind11_pybind11_python_link_helper_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_python_link_helper_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_python_link_helper_RES_DIRS_RELEASE )
set(pybind11_pybind11_python_link_helper_DEFINITIONS_RELEASE )
set(pybind11_pybind11_python_link_helper_OBJECTS_RELEASE )
set(pybind11_pybind11_python_link_helper_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_python_link_helper_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_python_link_helper_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_python_link_helper_LIBS_RELEASE )
set(pybind11_pybind11_python_link_helper_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_python_link_helper_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_python_link_helper_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_python_link_helper_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_python_link_helper_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_python_link_helper_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_python_link_helper_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_python_link_helper_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_python_link_helper_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_python_link_helper_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_python_link_helper_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_python_link_helper_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_python_link_helper_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_python_link_helper_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::module VARIABLES ############################################

set(pybind11_pybind11_module_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_module_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_module_BIN_DIRS_RELEASE )
set(pybind11_pybind11_module_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_module_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_module_RES_DIRS_RELEASE )
set(pybind11_pybind11_module_DEFINITIONS_RELEASE )
set(pybind11_pybind11_module_OBJECTS_RELEASE )
set(pybind11_pybind11_module_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_module_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_module_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_module_LIBS_RELEASE )
set(pybind11_pybind11_module_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_module_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_module_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_module_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_module_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_module_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_module_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_module_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_module_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_module_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_module_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_module_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_module_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_module_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::embed VARIABLES ############################################

set(pybind11_pybind11_embed_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_embed_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_embed_BIN_DIRS_RELEASE )
set(pybind11_pybind11_embed_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_embed_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_embed_RES_DIRS_RELEASE )
set(pybind11_pybind11_embed_DEFINITIONS_RELEASE )
set(pybind11_pybind11_embed_OBJECTS_RELEASE )
set(pybind11_pybind11_embed_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_embed_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_embed_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_embed_LIBS_RELEASE )
set(pybind11_pybind11_embed_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_embed_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_embed_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_embed_DEPENDENCIES_RELEASE pybind11::pybind11)
set(pybind11_pybind11_embed_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_embed_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_embed_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_embed_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_embed_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_embed_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_embed_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_embed_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_embed_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_embed_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::pybind11 VARIABLES ############################################

set(pybind11_pybind11_pybind11_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_pybind11_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_pybind11_BIN_DIRS_RELEASE )
set(pybind11_pybind11_pybind11_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_pybind11_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_pybind11_RES_DIRS_RELEASE )
set(pybind11_pybind11_pybind11_DEFINITIONS_RELEASE )
set(pybind11_pybind11_pybind11_OBJECTS_RELEASE )
set(pybind11_pybind11_pybind11_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_pybind11_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_pybind11_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_pybind11_LIBS_RELEASE )
set(pybind11_pybind11_pybind11_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_pybind11_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_pybind11_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_pybind11_DEPENDENCIES_RELEASE pybind11::headers)
set(pybind11_pybind11_pybind11_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_pybind11_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_pybind11_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_pybind11_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_pybind11_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_pybind11_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_pybind11_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_pybind11_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_pybind11_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_pybind11_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT pybind11::headers VARIABLES ############################################

set(pybind11_pybind11_headers_INCLUDE_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/include")
set(pybind11_pybind11_headers_LIB_DIRS_RELEASE "${pybind11_PACKAGE_FOLDER_RELEASE}/lib")
set(pybind11_pybind11_headers_BIN_DIRS_RELEASE )
set(pybind11_pybind11_headers_LIBRARY_TYPE_RELEASE UNKNOWN)
set(pybind11_pybind11_headers_IS_HOST_WINDOWS_RELEASE 0)
set(pybind11_pybind11_headers_RES_DIRS_RELEASE )
set(pybind11_pybind11_headers_DEFINITIONS_RELEASE )
set(pybind11_pybind11_headers_OBJECTS_RELEASE )
set(pybind11_pybind11_headers_COMPILE_DEFINITIONS_RELEASE )
set(pybind11_pybind11_headers_COMPILE_OPTIONS_C_RELEASE "")
set(pybind11_pybind11_headers_COMPILE_OPTIONS_CXX_RELEASE "")
set(pybind11_pybind11_headers_LIBS_RELEASE )
set(pybind11_pybind11_headers_SYSTEM_LIBS_RELEASE )
set(pybind11_pybind11_headers_FRAMEWORK_DIRS_RELEASE )
set(pybind11_pybind11_headers_FRAMEWORKS_RELEASE )
set(pybind11_pybind11_headers_DEPENDENCIES_RELEASE )
set(pybind11_pybind11_headers_SHARED_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_headers_EXE_LINK_FLAGS_RELEASE )
set(pybind11_pybind11_headers_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(pybind11_pybind11_headers_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${pybind11_pybind11_headers_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${pybind11_pybind11_headers_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${pybind11_pybind11_headers_EXE_LINK_FLAGS_RELEASE}>
)
set(pybind11_pybind11_headers_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${pybind11_pybind11_headers_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${pybind11_pybind11_headers_COMPILE_OPTIONS_C_RELEASE}>")