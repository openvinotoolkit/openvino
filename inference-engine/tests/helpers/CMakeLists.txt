# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME helpers)

file(GLOB HELPERS_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file (GLOB HELPERS_INCLUDES
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
      )

## Enable Models multiple search pathes
message("configuring file: ${CMAKE_CURRENT_BINARY_DIR}/test_model_repo.h")
configure_file(test_model_repo.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/test_model_repo.hpp @ONLY)


add_library(${TARGET_NAME} STATIC
            ${HELPERS_SRC}
            ${HELPERS_HEADERS})

# detecting regex support
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    target_compile_definitions(${TARGET_NAME} PUBLIC USE_BOOST_RE)

    debug_message(STATUS "Adding boost dependency")
    if (VERBOSE_BUILD)
        set(Boost_DEBUG on)
    endif ()
    find_package(Boost REQUIRED COMPONENTS regex)
    target_link_libraries(${TARGET_NAME} PUBLIC ${Boost_REGEX_LIBRARY})
    target_include_directories(${TARGET_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
endif()

if(MSVC)
    set(PUGI pugixml_mt)
else()
    set(PUGI pugixml)
endif()

if(WIN32)
    target_include_directories(${TARGET_NAME} PUBLIC "${IE_MAIN_SOURCE_DIR}/samples/common")
endif()

target_link_libraries(${TARGET_NAME} PUBLIC ${PUGI})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                                 $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
                                                 "${gtest_SOURCE_DIR}/include"
                                                 "${IE_MAIN_SOURCE_DIR}/src"
                                                 "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src"
                                                 "${gmock_SOURCE_DIR}/include"
                                         PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# TODO: eliminate dependency on samples
target_include_directories(${TARGET_NAME} PUBLIC 
    "${IE_MAIN_SOURCE_DIR}/samples/common")

target_compile_definitions(${TARGET_NAME} PUBLIC MODELS_PATH=\"${MODELS_PATH}\" DATA_PATH=\"${VALIDATION_SET}\")

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_PDB_NAME ${TARGET_NAME})
