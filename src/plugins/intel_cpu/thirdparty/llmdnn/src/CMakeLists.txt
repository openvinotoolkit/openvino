cmake_minimum_required(VERSION 3.18)
project(llmdnn)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(llmdnn STATIC ${SOURCE_FILES})
add_compile_definitions(DNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_TBB)
target_compile_definitions(llmdnn PRIVATE LLMDNN_EXPORT)
target_include_directories(llmdnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(llmdnn PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(TARGETS llmdnn DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
