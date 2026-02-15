# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_HETERO)
    return()
endif()

set(TARGET_NAME openvino_hetero_plugin)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "HETERO"
              PSEUDO_DEVICE
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR src/version.cpp
              ADD_CLANG_FORMAT)

ov_build_target_faster(${TARGET_NAME}
    UNITY
)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::pugixml)

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    if(BUILD_SHARED_LIBS)
        set(OBJ_NAME ${TARGET_NAME}_obj)

        add_library(${OBJ_NAME} OBJECT ${SOURCES} ${HEADERS})
        ov_link_system_libraries(${OBJ_NAME} PUBLIC openvino::pugixml)

        ov_add_version_defines(src/version.cpp ${OBJ_NAME})

        target_include_directories(${OBJ_NAME}
            PRIVATE
                $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

        ov_set_threading_interface_for(${OBJ_NAME})

        target_compile_definitions(${OBJ_NAME} PRIVATE USE_STATIC_IE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
    endif()

    add_subdirectory(tests)
endif()
