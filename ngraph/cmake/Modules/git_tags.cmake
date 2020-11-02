# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
# ******************************************************************************

function(NGRAPH_GET_CURRENT_HASH)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
        RESULT_VARIABLE result
        OUTPUT_VARIABLE HASH
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ERROR_QUIET)

    if(NOT HASH)
        return()
    endif()
    string(STRIP ${HASH} HASH)
    set(NGRAPH_CURRENT_HASH ${HASH} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_TAG_OF_CURRENT_HASH)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} show-ref
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE TAG_LIST
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ERROR_QUIET)

    NGRAPH_GET_CURRENT_HASH()

    if (NOT ${TAG_LIST} STREQUAL "")
        # first look for vX.Y.Z release tag
        string(REGEX MATCH "${NGRAPH_CURRENT_HASH}[\t ]+refs/tags/v([0-9?]+)\\.([0-9?]+)\\.([0-9?]+)$" TAG ${TAG_LIST})
        if ("${TAG}" STREQUAL "")
            # release tag not found so now look for vX.Y.Z-rc.N tag
            string(REGEX MATCH "${NGRAPH_CURRENT_HASH}[\t ]+refs/tags/v([0-9?]+)\\.([0-9?]+)\\.([0-9?]+)-(rc\\.[0-9?]+)$" TAG ${TAG_LIST})
        endif()
        set(STATUS ${TAG})
        if (NOT "${TAG}" STREQUAL "")
            string(REGEX REPLACE "${NGRAPH_CURRENT_HASH}[\t ]+refs/tags/(.*)" "\\1" FINAL_TAG ${TAG})
        endif()
    else()
        set(FINAL_TAG "")
    endif()
    set(NGRAPH_CURRENT_RELEASE_TAG ${FINAL_TAG} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_MOST_RECENT_TAG)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 --match v*.*.*
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE TAG
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ERROR_QUIET)

    if (NOT ${TAG} STREQUAL "")
        string(STRIP ${TAG} TAG)
    endif()
    set(NGRAPH_MOST_RECENT_RELEASE_TAG ${TAG} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_VERSION_LABEL)
    NGRAPH_GET_TAG_OF_CURRENT_HASH()
    set(NGRAPH_VERSION_LABEL ${NGRAPH_CURRENT_RELEASE_TAG} PARENT_SCOPE)
    if ("${NGRAPH_CURRENT_RELEASE_TAG}" STREQUAL "")
        NGRAPH_GET_CURRENT_HASH()
        NGRAPH_GET_MOST_RECENT_TAG()
        message(STATUS "Current hash ${NGRAPH_CURRENT_HASH}")
        string(SUBSTRING "${NGRAPH_CURRENT_HASH}" 0 7 HASH)
        if (NOT ${NGRAPH_MOST_RECENT_RELEASE_TAG} STREQUAL "")
            set(NGRAPH_VERSION_LABEL "${NGRAPH_MOST_RECENT_RELEASE_TAG}+${HASH}" PARENT_SCOPE)
        else()
            if(HASH)
                set(NGRAPH_VERSION_LABEL "v0.0.0+${HASH}" PARENT_SCOPE)
            else()
                # Not in a git repo
                if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/TAG)
                    # TAG file exists and TAG is assumed to be in the 'correct format', i.e. *.*.*
                    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/TAG NGRAPH_TAG)
                    string(STRIP ${NGRAPH_TAG} NGRAPH_TAG)
                    set(NGRAPH_VERSION_LABEL "${NGRAPH_TAG}" PARENT_SCOPE)
                else()
                    set(NGRAPH_VERSION_LABEL "v0.0.0+custom-build" PARENT_SCOPE)
                endif()
            endif()
        endif()
    endif()
endfunction()
