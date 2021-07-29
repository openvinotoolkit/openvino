CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

set(TARGET_NAME "cnpy")
add_library(cnpy STATIC  "cnpy.cpp")

if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-all")
    target_compile_options(${TARGET_NAME} PUBLIC -Wno-unused-variable)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC zlib)
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/..")

set_target_properties(cnpy PROPERTIES FOLDER thirdparty)