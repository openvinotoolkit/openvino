#===============================================================================
# Copyright 2016-2018 Intel Corporation
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

option(WITH_EXAMPLE "include examples" ON)
if (NOT WITH_EXAMPLE)
    return()
endif()

# propagate EXAMPLE specific flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_EXAMPLE_CCXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_EXAMPLE_CCXX_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(simple-net-c simple_net.c)
set_property(TARGET simple-net-c PROPERTY C_STANDARD 99)
target_link_libraries(simple-net-c ${LIB_NAME})
add_test(simple-net-c simple-net-c)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-net-c.vcxproj.user @ONLY)
    set_property(TEST simple-net-c PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-net-cpp simple_net.cpp)
set_property(TARGET simple-net-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(simple-net-cpp ${LIB_NAME})
add_test(simple-net-cpp simple-net-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-net-cpp.vcxproj.user @ONLY)
    set_property(TEST simple-net-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-training-net-c simple_training_net.c)
set_property(TARGET simple-training-net-c PROPERTY C_STANDARD 99)
if(WIN32)
target_link_libraries(simple-training-net-c ${LIB_NAME})
else()
target_link_libraries(simple-training-net-c ${LIB_NAME} m)
endif()
add_test(simple-training-net-c simple-training-net-c)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-training-net-c.vcxproj.user @ONLY)
    set_property(TEST simple-training-net-c PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-training-net-cpp simple_training_net.cpp)
set_property(TARGET simple-training-net-cpp PROPERTY CXX_STANDARD 11)
if(WIN32)
target_link_libraries(simple-training-net-cpp ${LIB_NAME})
else()
target_link_libraries(simple-training-net-cpp ${LIB_NAME} m)
endif()
add_test(simple-training-net-cpp simple-training-net-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-training-net-cpp.vcxproj.user @ONLY)
    set_property(TEST simple-training-net-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-net-int8-cpp simple_net_int8.cpp)
set_property(TARGET simple-net-int8-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(simple-net-int8-cpp ${LIB_NAME})
add_test(simple-net-int8-cpp simple-net-int8-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-net-int8-cpp.vcxproj.user @ONLY)
    set_property(TEST simple-net-int8-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

if (HAVE_MKL)
    add_executable(simple-rnn-cpp simple_rnn.cpp)
    set_property(TARGET simple-rnn-cpp PROPERTY CXX_STANDARD 11)
    target_link_libraries(simple-rnn-cpp ${LIB_NAME})
    add_test(simple-rnn-cpp simple-rnn-cpp)
    if(WIN32)
        configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
            simple-rnn-cpp.vcxproj.user @ONLY)
        set_property(TEST simple-rnn-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
    endif()
endif()
