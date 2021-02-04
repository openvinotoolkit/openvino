set(TARGET_NAME "samples_utils")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

include(compile_options)
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)

target_include_directories(${TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        # for os/windows/w_dirent.h
        "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(${TARGET_NAME}
    PUBLIC
        IE::inference_engine
        gflags
        vpu_common_lib)
