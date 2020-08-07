# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TARGET_NAME benchmark_app)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)


set(BENCHMARK_OPENCV_DEPENDENCIES imgcodecs)


find_package(OpenCV COMPONENTS ${BENCHMARK_OPENCV_DEPENDENCIES} QUIET)

if(NOT OpenCV_FOUND)
	message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
	return()
endif()


# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC})
if(HDR)
	source_group("include" FILES ${HDR})
endif()

add_executable(${TARGET_NAME} ${SRC} ${HDR})

target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)


target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    ${IE_MAIN_SOURCE_DIR}/samples/common
    ${IE_MAIN_SOURCE_DIR}/include
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_compile_options(${TARGET_NAME} PRIVATE
			"-Wall"
			)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES} inference_engine #${InferenceEngine_LIBRARIES}
                                               format_reader gflags)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME}
    FOLDER tools
)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
