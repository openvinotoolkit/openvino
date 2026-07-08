# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_util)

set(UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)

set(WIN32_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/os.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/parallel_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/win_file_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/win_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/win_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/win_mmap_object.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/win_shared_object_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/wstring_convert_util.cpp
)

set(UNIX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/lin_file_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/lin_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/lin_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/lin_mmap_object.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/lin_shared_object_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/parallel_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/wstring_convert_util.cpp
)

set(MONITOR_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/device_monitor.cpp
)

set(MONITOR_WIN32_SOURCES)

if(WIN32)
    if(TARGET IPF::ClientApi)
        list(APPEND MONITOR_WIN32_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/win/cpu_device_monitor.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/win/xpu_device_monitor.cpp
        )
    else()
        list(APPEND MONITOR_WIN32_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/win/cpu_device_monitor_stub.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/win/xpu_device_monitor_stub.cpp
        )
    endif()
endif()

set(MONITOR_UNIX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/lin/cpu_device_monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/lin/xpu_device_monitor.cpp
)

add_library(${TARGET_NAME} STATIC)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/file_util.cpp"
    PROPERTIES COMPILE_DEFINITIONS OpenVINO_VERSION="${OpenVINO_VERSION}")


target_sources(${TARGET_NAME}
    PRIVATE
        # Headers
        ${UTIL_INCLUDE_DIR}/openvino/pch/precomp_std.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/codec_xor.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/common_util.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/const_string.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/cpp_version.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/env_util.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/file_path.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/file_util.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/io.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/log.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/memory.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/mmap_object.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/native_stream.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/native_streambuf.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/os.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/ov_version.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/parallel_io.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/parallel_read_streambuf.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/pp.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/shared_object.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/util.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/variant_visitor.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/wstring_convert_util.hpp
        ${UTIL_INCLUDE_DIR}/openvino/util/xml_parse_utils.hpp

        # Sources
        ${CMAKE_CURRENT_SOURCE_DIR}/src/codec_xor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/env_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/file_path.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/file_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/log.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/native_stream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/native_streambuf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/parallel_read_streambuf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/xml_parse_utils.cpp
        ${MONITOR_COMMON_SOURCES}
        # Windows specific sources
        "$<$<BOOL:${WIN32}>:${WIN32_SOURCES}>"
        "$<$<BOOL:${WIN32}>:${MONITOR_WIN32_SOURCES}>"
         # Unix specific sources
        "$<$<BOOL:${UNIX}>:${UNIX_SOURCES}>"
        "$<$<BOOL:${UNIX}>:${MONITOR_UNIX_SOURCES}>"
)

add_library(openvino::util ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME util)

target_link_libraries(${TARGET_NAME}
    PRIVATE ${CMAKE_DL_LIBS}
    PUBLIC openvino::pugixml
           $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.1>>:stdc++fs>
           $<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:c++fs>)

if (WIN32)
    set(MONITORS_WIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/monitors/win/)
    target_include_directories(${TARGET_NAME} PRIVATE $<BUILD_INTERFACE:${MONITORS_WIN_INCLUDE_DIR}>)

    target_link_libraries(${TARGET_NAME} PRIVATE Shlwapi)

    # IPF (Intel Platform Framework) SDK for device monitoring
    if(TARGET IPF::ClientApi)
        target_link_libraries(${TARGET_NAME} PRIVATE IPF::ClientApi)
    endif()
endif()

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${UTIL_INCLUDE_DIR}>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES ${UTIL_INCLUDE_DIR})

set_target_properties(${TARGET_NAME} PROPERTIES PCH_CORE_HEADER_PATH "${UTIL_INCLUDE_DIR}/openvino/pch/precomp_std.hpp")

ov_build_target_faster(${TARGET_NAME} PCH)

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${UTIL_INCLUDE_DIR}/")
