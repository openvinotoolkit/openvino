# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET py_ov_frontends)
    add_custom_target(py_ov_frontends)
endif()

function(frontend_module TARGET FRAMEWORK INSTALL_COMPONENT)
    set(TARGET_NAME ${TARGET})

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/${FRAMEWORK})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/${FRAMEWORK})
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/${FRAMEWORK})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/${FRAMEWORK})
    set(PYTHON_BRIDGE_CPACK_PATH "${OV_CPACK_PYTHONDIR}")

    file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

    # create target

    pybind11_add_module(${TARGET_NAME} MODULE NO_EXTRAS ${SOURCES})

    add_dependencies(${TARGET_NAME} pyopenvino)
    add_dependencies(py_ov_frontends ${TARGET_NAME})

    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                      "${OpenVINOPython_SOURCE_DIR}/src/pyopenvino/utils/")
    target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::frontend::${FRAMEWORK})

    set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

    # Compatibility with python 2.7 which has deprecated "register" specifier
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
    endif()

    # perform copy
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy  ${OpenVINOPython_SOURCE_DIR}/src/openvino/frontend/${FRAMEWORK}/__init__.py
                                              ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py)

    install(TARGETS ${TARGET_NAME}
            DESTINATION ${OV_CPACK_PYTHONDIR}/${pyversion}/openvino/frontend/${FRAMEWORK}
            COMPONENT ${INSTALL_COMPONENT})
endfunction()
