cmake_minimum_required(VERSION 3.9)

set (TARGET_NAME "inference_engine_node_api")


set(CMAKE_CXX_STANDARD 11)

if(DEFINED IE_MAIN_SOURCE_DIR)
    find_package(InferenceEngine REQUIRED)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${InferenceEngine_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/node_modules/node-addon-api
                    ${CMAKE_CURRENT_SOURCE_DIR}/node_modules/node-addon-api/src
                    )

file(GLOB SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        )

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/node_api")

target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES})
