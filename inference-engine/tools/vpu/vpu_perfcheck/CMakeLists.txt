#
# Copyright (C) 2018-2020 Intel Corporation
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
#

disable_deprecated_warnings()

function(add_perfcheck_target TARGET_NAME PLUGIN_NAME)
    find_package(Threads REQUIRED)

    file(GLOB SOURCES *.cpp)

    add_executable(${TARGET_NAME} ${SOURCES})

    # TODO: enable some day and fix all warnings
#    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#        target_compile_options(${TARGET_NAME}
#            PRIVATE
#                "-Wall")
#    endif()

    target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
            "${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include"
            "${IE_MAIN_SOURCE_DIR}/samples/common/samples"
            "${IE_MAIN_SOURCE_DIR}/samples/common/format_reader")

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            inference_engine format_reader
            Threads::Threads)

    add_dependencies(${TARGET_NAME}
        ${PLUGIN_NAME} ${ARGN})

    set_target_properties(${TARGET_NAME} PROPERTIES
        COMPILE_PDB_NAME ${TARGET_NAME})

    add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endfunction()

if(ENABLE_MYRIAD)
    add_perfcheck_target(myriad_perfcheck myriadPlugin)

    install(TARGETS myriad_perfcheck
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT myriad)
endif()
