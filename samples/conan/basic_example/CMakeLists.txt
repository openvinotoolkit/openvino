cmake_minimum_required(VERSION 3.20)
project(conan_basic_example)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(SOURCE ${CMAKE_SOURCE_DIR}/main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE})

conan_target_link_libraries(${PROJECT_NAME})

# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
