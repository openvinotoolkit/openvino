# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_reference")

set(REF_IMPL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${REF_IMPL_INCLUDE_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::reference ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME reference)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp")

ov_set_threading_interface_for(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME} PRIVATE XBYAK_NO_OP_NAMES XBYAK64)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${REF_IMPL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OV_CORE_DEV_API_PATH}>
    $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<BUILD_INTERFACE:$<$<TARGET_EXISTS:xbyak::xbyak>:$<TARGET_PROPERTY:xbyak::xbyak,INTERFACE_INCLUDE_DIRECTORIES>>>)

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads openvino::core::dev)

#find jpeg vars

if(JPEG_FOUND)
    include_directories(${JPEG_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE ${JPEG_LIBRARIES})
    add_compile_definitions(WITH_JPEG)
    message(STATUS "JPEG is found, JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}, JPEG_LIBRARIES=${JPEG_LIBRARIES}") 
else()
    message(STATUS "JPEG is not found, disable libjpeg support") 
endif()


#find png vars
if(PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE ${PNG_LIBRARIES})
    add_compile_definitions(WITH_PNG)
    message(STATUS "PNG is found, PNG_INCLUDE_DIR=${PNG_INCLUDE_DIR}, PNG_LIBRARIES=${PNG_LIBRARIES}") 
else()
    message(STATUS "PNG is not found, disable libpng support") 
endif()


#find gif vars
if(GIF_FOUND)
    include_directories(${GIF_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE ${GIF_LIBRARIES})
    add_compile_definitions(WITH_GIF)
    message(STATUS "GIF is found, GIF_INCLUDE_DIR=${GIF_INCLUDE_DIR}, GIF_LIBRARIES=${GIF_LIBRARIES}") 
else()
    message(STATUS "GIF is not found, build libgif in thirdparty") 
endif()

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::reference
                                    INSTALL_INCLUDE_DIRECTORIES "${REF_IMPL_INCLUDE_DIR}/")
