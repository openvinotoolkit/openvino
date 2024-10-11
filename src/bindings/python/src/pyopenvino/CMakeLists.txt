# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project (pyopenvino)

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(OpenVINODeveloperPackage REQUIRED)
endif()

# Python3_VERSION_MAJOR and Python3_VERSION_MINOR are defined by FindPython3
set(pyversion python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(OV_GENERATOR_MULTI_CONFIG)
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python/openvino)
else()
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/openvino)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})

# compile and linker options

if(OV_COMPILER_IS_APPLECLANG)
    add_link_options(-stdlib=libc++)
elseif(CMAKE_COMPILER_IS_GNUCXX)
    # WA for GCC 7.5 "PYBIND11_NOINLINE inline" warning
    ov_add_compiler_flags(-Wno-error=attributes)
endif()

if(ENABLE_TESTS)
    add_subdirectory(test_utils)
endif()

if(TARGET openvino::frontend::onnx)
    add_subdirectory(frontend/onnx)
endif()

if(TARGET openvino::frontend::tensorflow)
    add_subdirectory(frontend/tensorflow)
endif()

if(TARGET openvino::frontend::paddle)
    add_subdirectory(frontend/paddle)
endif()

if(TARGET openvino::frontend::pytorch)
    add_subdirectory(frontend/pytorch)
endif()

if(TARGET openvino::frontend::jax)
    add_subdirectory(frontend/jax)
endif()

# create target

file(GLOB_RECURSE SOURCES core/*.cpp experimental/*.cpp graph/*.cpp frontend/*.cpp utils/*.cpp pyopenvino.cpp)
list(FILTER SOURCES EXCLUDE REGEX ".*(frontend/(onnx|tensorflow|paddle|pytorch|jax))/*")

pybind11_add_module(${PROJECT_NAME} MODULE NO_EXTRAS ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(${PROJECT_NAME} PRIVATE openvino::core::dev openvino::runtime openvino::offline_transformations)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
    OUTPUT_NAME "_pyopenvino")

ov_add_version_defines(pyopenvino.cpp ${PROJECT_NAME})

if(OV_GENERATOR_MULTI_CONFIG)
    string(APPEND _cmd_echo
        "$<$<CONFIG:Debug>:"
            "${CMAKE_COMMAND};-E;cmake_echo_color;--red;\"OpenVINO;Python;API;cannot;be;built;for;'Debug'\""
        ">")
    string(APPEND cmd_error
        "$<$<CONFIG:Debug>:"
            "${CMAKE_COMMAND};-E;false"
        ">")

    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
        COMMAND "${_cmd_echo}"
        COMMAND "${cmd_error}"
        COMMAND_EXPAND_LISTS)
endif()

# perform copy
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenVINOPython_SOURCE_DIR}/src/openvino ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenVINOPython_SOURCE_DIR}/requirements.txt ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../requirements.txt
        )

ov_python_minimal_api(${PROJECT_NAME})
ov_add_clang_format_target(${PROJECT_NAME}_clang FOR_TARGETS ${PROJECT_NAME})

# install steps

ov_cpack_add_component(${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion}
                        HIDDEN)

install(DIRECTORY ${OpenVINOPython_SOURCE_DIR}/src/openvino
        DESTINATION ${OV_CPACK_PYTHONDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion}
        ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL}
        USE_SOURCE_PERMISSIONS
        PATTERN "test_utils" EXCLUDE
        PATTERN "torchvision/requirements.txt" EXCLUDE)

install(TARGETS ${PROJECT_NAME}
        DESTINATION ${OV_CPACK_PYTHONDIR}/openvino
        COMPONENT ${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion}
        ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})

ov_set_install_rpath(${PROJECT_NAME} ${OV_CPACK_PYTHONDIR}/openvino
                     # path to OpenVINO C++ libraries
                     ${OV_CPACK_RUNTIMEDIR}
                     # pyopenvino also depends on TBB because of:
                     # pyopenvino => openvino::offline_transformations => TBB optimized openvino::reference
                     ${TBB_LIB_INSTALL_DIR})

ov_cpack_add_component(${OV_CPACK_COMP_OPENVINO_REQ_FILES} HIDDEN)

install(FILES ${OpenVINOPython_SOURCE_DIR}/requirements.txt
        DESTINATION ${OV_CPACK_PYTHONDIR}
        COMPONENT ${OV_CPACK_COMP_OPENVINO_REQ_FILES}
        ${OV_CPACK_COMP_OPENVINO_REQ_FILES_EXCLUDE_ALL})

install(FILES ${OpenVINOPython_SOURCE_DIR}/src/openvino/preprocess/torchvision/requirements.txt
        DESTINATION ${OV_CPACK_PYTHONDIR}/openvino/preprocess/torchvision
        COMPONENT ${OV_CPACK_COMP_OPENVINO_REQ_FILES}
        ${OV_CPACK_COMP_OPENVINO_REQ_FILES_EXCLUDE_ALL})

install(DIRECTORY ${OpenVINOPython_SOURCE_DIR}/tests
        DESTINATION tests/${PROJECT_NAME}
        COMPONENT tests
        EXCLUDE_FROM_ALL)

if(TARGET ie_wheel)
    add_dependencies(ie_wheel ${PROJECT_NAME})
endif()
