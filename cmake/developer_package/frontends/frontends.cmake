# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FRONTEND_INSTALL_INCLUDE "runtime/include/")
set(FRONTEND_NAME_PREFIX "openvino_")
set(FRONTEND_NAME_SUFFIX "_frontend")

set(FRONTEND_NAMES "" CACHE INTERNAL "")

if(NOT TARGET ov_frontends)
    add_custom_target(ov_frontends)
endif()

#
# ov_target_link_frontends(<TARGET_NAME>)
#
function(ov_target_link_frontends TARGET_NAME)
    if(BUILD_SHARED_LIBS)
        return()
    endif()

    foreach(name IN LISTS FRONTEND_NAMES)
        set(frontend_target_name "${FRONTEND_NAME_PREFIX}${name}${FRONTEND_NAME_SUFFIX}")
        target_link_libraries(${TARGET_NAME} PRIVATE ${frontend_target_name})
    endforeach()
endfunction()

#
# ov_generate_frontends_hpp()
#
function(ov_generate_frontends_hpp)
    if(BUILD_SHARED_LIBS)
        return()
    endif()

    # add frontends to libraries including ov_frontends.hpp
    ov_target_link_frontends(openvino)

    set(ov_frontends_hpp "${CMAKE_BINARY_DIR}/src/frontends/common/src/ov_frontends.hpp")
    set(frontends_hpp_in "${IEDevScripts_DIR}/frontends/ov_frontends.hpp.in")

    add_custom_command(OUTPUT "${ov_frontends_hpp}"
                       COMMAND
                        "${CMAKE_COMMAND}"
                        -D "OV_FRONTENDS_HPP_HEADER_IN=${frontends_hpp_in}"
                        -D "OV_FRONTENDS_HPP_HEADER=${ov_frontends_hpp}"
                        -D "FRONTEND_NAMES=${FRONTEND_NAMES}"
                        -P "${IEDevScripts_DIR}/frontends/create_frontends_hpp.cmake"
                       DEPENDS
                         "${frontends_hpp_in}"
                         "${IEDevScripts_DIR}/frontends/create_frontends_hpp.cmake"
                       COMMENT
                         "Generate ov_frontends.hpp for static build"
                       VERBATIM)

    # for some reason dependency on source files does not work
    # so, we have to use explicit target and make it dependency for frontend_common
    add_custom_target(_ov_frontends_hpp DEPENDS ${ov_frontends_hpp})
    add_dependencies(frontend_common_obj _ov_frontends_hpp)

    # add dependency for object files
    get_target_property(sources frontend_common_obj SOURCES)
    foreach(source IN LISTS sources)
        if("${source}" MATCHES "\\$\\<TARGET_OBJECTS\\:([A-Za-z0-9_]*)\\>")
            # object library
            set(obj_library ${CMAKE_MATCH_1})
            get_target_property(obj_sources ${obj_library} SOURCES)
            list(APPEND all_sources ${obj_sources})
        else()
            # usual source
            list(APPEND all_sources ${source})
        endif()
    endforeach()

    # add dependency on header file generation for all inference_engine source files
    set_source_files_properties(${all_sources} PROPERTIES OBJECT_DEPENDS ${ov_frontends_hpp})
endfunction()

unset(protobuf_lite_installed CACHE)
unset(protobuf_installed CACHE)

#
# ov_add_frontend(NAME <IR|ONNX|...>
#                 FILEDESCRIPTION <description> # used on Windows to describe DLL file
#                 [LINKABLE_FRONTEND] # whether we can use FE API directly or via FEM only
#                 [SKIP_INSTALL] # private frontend, not for end users
#                 [PROTOBUF_LITE] # requires only libprotobuf-lite
#                 [SKIP_NCC_STYLE] # use custom NCC rules
#                 [LINK_LIBRARIES <lib1 lib2 ...>])
#
macro(ov_add_frontend)
    set(options LINKABLE_FRONTEND SHUTDOWN_PROTOBUF PROTOBUF_LITE SKIP_NCC_STYLE SKIP_INSTALL)
    set(oneValueArgs NAME FILEDESCRIPTION)
    set(multiValueArgs LINK_LIBRARIES PROTO_FILES)
    cmake_parse_arguments(OV_FRONTEND "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(prop NAME FILEDESCRIPTION)
        if(NOT DEFINED OV_FRONTEND_${prop})
            message(FATAL_ERROR "Frontend ${prop} property is not defined")
        endif()
    endforeach()

    set(TARGET_NAME "${FRONTEND_NAME_PREFIX}${OV_FRONTEND_NAME}${FRONTEND_NAME_SUFFIX}")

    list(APPEND FRONTEND_NAMES ${OV_FRONTEND_NAME})
    set(FRONTEND_NAMES "${FRONTEND_NAMES}" CACHE INTERNAL "" FORCE)

    file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
    if (WIN32)
        # Remove linux specific files
        file(GLOB_RECURSE LIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.hpp)
        list(REMOVE_ITEM LIBRARY_SRC "${LIN_FILES}")
    else()
        # Remove windows specific files
        file(GLOB_RECURSE WIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.hpp)
        list(REMOVE_ITEM LIBRARY_SRC "${WIN_FILES}")
    endif()
    file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
    file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

    set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj

    source_group("src" FILES ${LIBRARY_SRC})
    source_group("include" FILES ${LIBRARY_HEADERS})
    source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

    # Generate protobuf file on build time for each '.proto' file in src/proto
    file(GLOB proto_files ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.proto)

    foreach(INFILE IN LISTS proto_files)
        get_filename_component(FILE_DIR ${INFILE} DIRECTORY)
        get_filename_component(FILE_WE ${INFILE} NAME_WE)
        set(OUTPUT_PB_SRC ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}.pb.cc)
        set(OUTPUT_PB_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}.pb.h)
        set(GENERATED_PROTO ${INFILE})
        add_custom_command(
                OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
                COMMAND ${PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${FILE_DIR} ${FILE_WE}.proto
                DEPENDS ${PROTOC_EXECUTABLE} ${GENERATED_PROTO}
                COMMENT "Running C++ protocol buffer compiler (${PROTOC_EXECUTABLE}) on ${GENERATED_PROTO}"
                VERBATIM
                COMMAND_EXPAND_LISTS)
        list(APPEND PROTO_SRCS "${OUTPUT_PB_SRC}")
        list(APPEND PROTO_HDRS "${OUTPUT_PB_HEADER}")
    endforeach()

    # Disable all warnings for generated code
    set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES COMPILE_OPTIONS -w GENERATED TRUE)

    # Create library
    add_library(${TARGET_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})

    if(OV_FRONTEND_LINKABLE_FRONTEND)
        # create beautiful alias
        add_library(openvino::frontend::${OV_FRONTEND_NAME} ALIAS ${TARGET_NAME})
    endif()

    # Shutdown protobuf when unloading the front dynamic library
    if(OV_FRONTEND_SHUTDOWN_PROTOBUF AND BUILD_SHARED_LIBS)
        target_link_libraries(${TARGET_NAME} PRIVATE ov_protobuf_shutdown)
    endif()

    if(NOT BUILD_SHARED_LIBS)
        # override default function names
        target_compile_definitions(${TARGET_NAME} PRIVATE
            "-DGetFrontEndData=GetFrontEndData${OV_FRONTEND_NAME}"
            "-DGetAPIVersion=GetAPIVersion${OV_FRONTEND_NAME}")
    endif()

    if(OV_FRONTEND_SKIP_NCC_STYLE)
        # frontend's CMakeLists.txt must define its own custom 'ov_ncc_naming_style' step
    else()
        ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                            SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
                            ADDITIONAL_INCLUDE_DIRECTORIES
                                $<TARGET_PROPERTY:frontend_common::static,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    target_include_directories(${TARGET_NAME}
            PUBLIC
                $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_BINARY_DIR})

    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION ${OV_FRONTEND_FILEDESCRIPTION})

    ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

    target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OV_FRONTEND_LINK_LIBRARIES})

    # WA for TF frontends which always requires protobuf (not protobuf-lite)
    # if TF FE is built in static mode, use protobuf for all other FEs
    if(FORCE_FRONTENDS_USE_PROTOBUF)
        set(OV_FRONTEND_PROTOBUF_LITE OFF)
    endif()

    if(proto_files)
        if(OV_FRONTEND_PROTOBUF_LITE)
            if(NOT protobuf_lite_installed)
                ov_install_static_lib(${Protobuf_LITE_LIBRARIES} core)
                set(protobuf_lite_installed ON CACHE INTERNAL "" FORCE)
            endif()
            link_system_libraries(${TARGET_NAME} PRIVATE ${Protobuf_LITE_LIBRARIES})
        else()
            if(NOT protobuf_installed)
                ov_install_static_lib(${Protobuf_LIBRARIES} core)
                set(protobuf_installed ON CACHE INTERNAL "" FORCE)
            endif()
            link_system_libraries(${TARGET_NAME} PRIVATE ${Protobuf_LIBRARIES})
        endif()

        # prptobuf generated code emits -Wsuggest-override error
        if(SUGGEST_OVERRIDE_SUPPORTED)
            target_compile_options(${TARGET_NAME} PRIVATE -Wno-suggest-override)
        endif()
    endif()

    add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                            EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

    add_dependencies(ov_frontends ${TARGET_NAME})

    if(NOT OV_FRONTEND_SKIP_INSTALL)
        if(BUILD_SHARED_LIBS)
            if(OV_FRONTEND_LINKABLE_FRONTEND)
                set(export_set EXPORT OpenVINOTargets)
            endif()
            install(TARGETS ${TARGET_NAME} ${export_set}
                    RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
                    ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
                    LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)
        else()
            ov_install_static_lib(${TARGET_NAME} core)
        endif()

        if(OV_FRONTEND_LINKABLE_FRONTEND)
            # install library development files
            install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/openvino
                    DESTINATION ${FRONTEND_INSTALL_INCLUDE}/
                    COMPONENT core_dev
                    FILES_MATCHING PATTERN "*.hpp")

            set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::${OV_FRONTEND_NAME})
            export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
                   APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
        endif()
    else()
        # skipped frontend has to be installed in static libraries case
        ov_install_static_lib(${TARGET_NAME} core)
    endif()
endmacro()
