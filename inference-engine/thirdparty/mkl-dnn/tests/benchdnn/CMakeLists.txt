#===============================================================================
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    )
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/ip
    ${CMAKE_CURRENT_SOURCE_DIR}/reorder
    )

option(BENCHDNN_USE_RDPMC
    "enables rdpms counter to report precise cpu frequency.
     CAUTION: may not work on all cpus (hence disable by default)"
    OFF) # disabled by default
if(BENCHDNN_USE_RDPMC)
    add_definitions(-DBENCHDNN_USE_RDPMC)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    if(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qprec-div")
    elseif(UNIX OR APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -prec-div")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model precise")
    endif()
endif()

add_executable(benchdnn ${SOURCES})
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        benchdnn.vcxproj.user @ONLY)
    set_property(TARGET benchdnn PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()
set_property(TARGET benchdnn PROPERTY CXX_STANDARD 11)
target_link_libraries(benchdnn ${LIB_NAME} ${EXTRA_LIBS})
if(UNIX AND NOT APPLE)
    find_library(LIBRT rt)
    if(LIBRT)
        target_link_libraries(benchdnn ${LIBRT})
    endif()
endif()
# add_test(benchdnn benchdnn)

file(COPY inputs DESTINATION .)
add_custom_target(test_conv
    COMMAND benchdnn --conv --batch=inputs/test_conv_all
    DEPENDS benchdnn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        test_conv.vcxproj.user @ONLY)
    set_property(TARGET test_conv PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()
add_custom_target(test_benchdnn_other
    COMMAND benchdnn --reorder
    DEPENDS benchdnn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        test_benchdnn_other.vcxproj.user @ONLY)
    set_property(TARGET test_benchdnn_other PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()
add_custom_target(test_conv_regression
    COMMAND benchdnn --conv --batch=inputs/test_conv_regression
    DEPENDS benchdnn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        test_conv_regression.vcxproj.user @ONLY)
    set_property(TARGET test_conv_regression PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()
