# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_util)

set(UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${UTIL_INCLUDE_DIR}/*.hpp)

if (WIN32)
    # Remove linux specific files
    file(GLOB_RECURSE LIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.hpp)
    list(REMOVE_ITEM LIBRARY_SRC ${LIN_FILES})
else()
    # Remove windows specific files
    file(GLOB_RECURSE WIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.hpp)
    list(REMOVE_ITEM LIBRARY_SRC ${WIN_FILES})
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/file_util.cpp"
    PROPERTIES COMPILE_DEFINITIONS OpenVINO_VERSION="${OpenVINO_VERSION}")

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::util ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME util)

file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    "#include <experimental/filesystem>\nint main(int argc, char ** argv) {\n  std::experimental::filesystem::path p(argv[0]);\n  return p.string().length();\n}"
  )
  try_compile(
    STD_FS_NO_LIB_NEEDED ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    COMPILE_DEFINITIONS -std=c++11)
  try_compile(
    STD_FS_NEEDS_STDCXXFS ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    COMPILE_DEFINITIONS -std=c++11
    LINK_LIBRARIES stdc++fs)
  try_compile(
    STD_FS_NEEDS_CXXFS ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    COMPILE_DEFINITIONS -std=c++11
    LINK_LIBRARIES c++fs)

if(MSVC)
  message(INFO "MSVC - No explicit filesystem linker setting required.")
  set(STD_FS_LIB "")
elseif(STD_FS_NEEDS_STDCXXFS)
  message(INFO "STD_FS_NEEDS_STDCXXFS - Add explicit filesystem linker setting: 'stdc++fs'.")
  set(STD_FS_LIB stdc++fs)
elseif(STD_FS_NEEDS_CXXFS)
  message(INFO "STD_FS_NEEDS_CXXFS - Add explicit filesystem linker setting: 'c++fs'.")
  set(STD_FS_LIB c++fs)
elseif(STD_FS_NO_LIB_NEEDED)
  message(INFO "STD_FS_NO_LIB_NEEDED - No explicit filesystem linker setting required.")
  set(STD_FS_LIB "")
else()
  message(WARNING "Unknown C++ build setup - No explicit filesystem linker setting set")
  set(STD_FS_LIB "")
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS} PUBLIC openvino::pugixml  ${STD_FS_LIB} )
if (WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE Shlwapi)
endif()
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${UTIL_INCLUDE_DIR}>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES ${UTIL_INCLUDE_DIR})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${UTIL_INCLUDE_DIR}/")
