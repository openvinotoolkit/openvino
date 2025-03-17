# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "openvino_runtime")

file (GLOB LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dev/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dev/preprocessing/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dev/threading/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/threading/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_interfaces/interface/*.cpp
      )

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
      )

if(WIN32)
    file (GLOB LIBRARY_SRC
         ${LIBRARY_SRC}
         ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.cpp)
    file (GLOB LIBRARY_HEADERS
         ${LIBRARY_HEADERS}
         ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.hpp)
elseif(APPLE)
    file (GLOB LIBRARY_SRC
        ${LIBRARY_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/os/mac/*.cpp)
    file (GLOB LIBRARY_HEADERS
        ${LIBRARY_HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/os/mac/*.hpp)
elseif(NOT EMSCRIPTEN)
    file (GLOB LIBRARY_SRC
          ${LIBRARY_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.cpp)
    file (GLOB LIBRARY_HEADERS
         ${LIBRARY_HEADERS}
         ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.hpp)
endif()

set (PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file (GLOB_RECURSE PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/*.hpp
       ${PUBLIC_HEADERS_DIR}/*.h
      )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})


if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

ov_build_target_faster(${TARGET_NAME}_obj
    UNITY PCH PRIVATE "src/precomp.hpp"
)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE
    IMPLEMENT_OPENVINO_RUNTIME_API
    $<$<TARGET_EXISTS:openvino_proxy_plugin_obj>:PROXY_PLUGIN_ENABLED>)

if(ENABLE_DEBUG_CAPS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC ENABLE_DEBUG_CAPS)
endif()

target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE
    $<TARGET_PROPERTY:openvino::pugixml,INTERFACE_INCLUDE_DIRECTORIES>
    $<$<TARGET_EXISTS:xbyak::xbyak>:$<TARGET_PROPERTY:xbyak::xbyak,INTERFACE_INCLUDE_DIRECTORIES>>)

target_include_directories(${TARGET_NAME}_obj PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
    $<$<TARGET_EXISTS:openvino_proxy_plugin_obj>:$<TARGET_PROPERTY:openvino_proxy_plugin_obj,INTERFACE_INCLUDE_DIRECTORIES>>
    # for ov_plugins.hpp
    $<IF:$<AND:$<BOOL:${OV_GENERATOR_MULTI_CONFIG}>,$<VERSION_GREATER_EQUAL:${CMAKE_VERSION},3.20>>,${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>,${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::itt openvino::util openvino::core::dev nlohmann_json::nlohmann_json)
ov_mark_target_as_cc(${TARGET_NAME}_obj)

# OpenVINO Runtime is public API => need to mark this library as important for ABI free
ov_abi_free_target(${TARGET_NAME}_obj)

ov_set_threading_interface_for(${TARGET_NAME}_obj)
if (TBBBIND_2_5_FOUND)
    target_link_libraries(${TARGET_NAME}_obj PRIVATE ${TBBBIND_2_5_IMPORTED_TARGETS})
endif()

# Create library file from object library

add_library(${TARGET_NAME} INTERFACE)
target_link_libraries(${TARGET_NAME} INTERFACE openvino::runtime)
target_include_directories(${TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}_obj
                    SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/openvino"
                    ADDITIONAL_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            $<$<TARGET_EXISTS:openvino_proxy_plugin_obj>:$<TARGET_OBJECTS:openvino_proxy_plugin_obj>>
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

ov_set_threading_interface_for(${TARGET_NAME}_s)
if (TBBBIND_2_5_FOUND)
    target_link_libraries(${TARGET_NAME}_s PRIVATE ${TBBBIND_2_5_IMPORTED_TARGETS})
endif()

target_include_directories(${TARGET_NAME}_s PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${TARGET_NAME}_s PRIVATE openvino::itt ${CMAKE_DL_LIBS}
    openvino::runtime::dev openvino::pugixml)

target_compile_definitions(${TARGET_NAME}_s PUBLIC USE_STATIC_IE)

if(ENABLE_DEBUG_CAPS)
    target_compile_definitions(${TARGET_NAME}_s PUBLIC ENABLE_DEBUG_CAPS)
endif()

set_target_properties(${TARGET_NAME}_s PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# LTO

set_target_properties(${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Export for developer package

ov_developer_package_export_targets(TARGET ${TARGET_NAME}_s)

# Install static libraries for case BUILD_SHARED_LIBS=OFF

install(DIRECTORY "${PUBLIC_HEADERS_DIR}/"
        DESTINATION ${OV_CPACK_INCLUDEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
