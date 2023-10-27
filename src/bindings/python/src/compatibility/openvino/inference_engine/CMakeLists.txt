# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ie_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ie_api.pyx
                  ${CMAKE_CURRENT_SOURCE_DIR}/*.pxd
                  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

file(GLOB PYX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx)
set_source_files_properties(${PYX_SOURCES} PROPERTIES CYTHON_IS_CXX ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 'argument': conversion from 'size_t' to 'int', possible loss of data
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4244)
    ov_add_compiler_flags(/wd4551)
endif()
if(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-undef)
    if(OV_COMPILER_IS_CLANG)
        ov_add_compiler_flags(-Wno-parentheses-equality)
    endif()
endif()
if(UNUSED_BUT_SET_VARIABLE_SUPPORTED)
    ov_add_compiler_flags(-Wno-unused-but-set-variable)
endif()

# create target

cython_add_module(${TARGET_NAME} ${SOURCES})
ov_python_disable_intel_warnings(${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

set(INSTALLED_TARGETS ${TARGET_NAME})
list(REMOVE_ITEM PYX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ie_api.pyx")

foreach(PYX_FILE IN LISTS PYX_SOURCES)
    get_filename_component(PYX_NAME "${PYX_FILE}" NAME_WE)
    cython_add_module(${PYX_NAME} ${PYX_FILE})
    ov_python_disable_intel_warnings(${PYX_NAME})
    add_dependencies(${TARGET_NAME} ${PYX_NAME})
    target_include_directories(${PYX_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${PYX_NAME} PRIVATE openvino::runtime)
    list(APPEND INSTALLED_TARGETS ${PYX_NAME})
    ov_python_minimal_api(${PYX_NAME})
    set_target_properties(${PYX_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endforeach()

if(COMMAND ov_add_vs_version_file)
    foreach(target IN LISTS INSTALLED_TARGETS)
        ov_add_vs_version_file(NAME ${target}
                               FILEDESCRIPTION "Inference Engine Python library")
    endforeach()
endif()

function(python_ov_disable_deprecated_warnings)
    ov_disable_deprecated_warnings()
    set(pyx_file "${CMAKE_CURRENT_BINARY_DIR}/ie_api.cxx" "${CMAKE_CURRENT_BINARY_DIR}/constants.cxx")
    set_source_files_properties(${pyx_file} PROPERTIES COMPILE_OPTIONS ${ov_c_cxx_deprecated})
endfunction()

python_ov_disable_deprecated_warnings()
ov_python_minimal_api(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
)

foreach(target IN LISTS INSTALLED_TARGETS)
    ov_set_apple_rpath(${target} ${OV_CPACK_PYTHONDIR}/openvino/inference_engine ${OV_CPACK_RUNTIMEDIR})
endforeach()

# install

install(TARGETS ${INSTALLED_TARGETS}
        RUNTIME DESTINATION ${OV_CPACK_PYTHONDIR}/openvino/inference_engine
        COMPONENT ${PYTHON_COMPONENT} ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL}
        LIBRARY DESTINATION ${OV_CPACK_PYTHONDIR}/openvino/inference_engine
        COMPONENT ${PYTHON_COMPONENT} ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})

install(PROGRAMS __init__.py
        DESTINATION ${OV_CPACK_PYTHONDIR}/openvino/inference_engine
        COMPONENT ${PYTHON_COMPONENT}
        ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                           EXCLUDE_PATTERNS ".*\\.cxx;.*\\.pxd;.*\\.pyx")
