# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_DOCKER)
    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
endif()

cmake_policy(SET CMP0054 NEW)

find_package(Git REQUIRED)

set(MODELS_LST "")
set(MODELS_LST_TO_FETCH "")

function (add_models_repo add_to_fetcher model_name)
    list(LENGTH ARGV add_models_args)
    if (add_models_args EQUAL 3)
        list(GET ARGV 2 branch_name)
    else()
        set(branch_name ${MODELS_BRANCH})
    endif()
    if (add_to_fetcher)
        set(model_name "${model_name}:${branch_name}")
        list(APPEND MODELS_LST_TO_FETCH ${model_name})
    endif()

    list(APPEND MODELS_LST ${model_name})

    set(MODELS_LST_TO_FETCH ${MODELS_LST_TO_FETCH} PARENT_SCOPE)
    set(MODELS_LST ${MODELS_LST} PARENT_SCOPE)
endfunction()

function(add_lfs_repo name prefix url tag)
    if(TARGET ${name})
        return()
    endif()

    ExternalProject_Add(${name}
        PREFIX ${prefix}
        GIT_REPOSITORY ${url}
        GIT_TAG ${tag}
        GIT_CONFIG "http.sslverify=false"
        GIT_PROGRESS 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON)

    execute_process(
        COMMAND ${GIT_EXECUTABLE} lfs install --local --force
        WORKING_DIRECTORY ${prefix}/src/${name}
        OUTPUT_VARIABLE lfs_output
        RESULT_VARIABLE lfs_var)
    if(lfs_var)
        message(FATAL_ERROR [=[
            Failed to setup Git LFS: ${lfs_output}
            Git lfs must be installed in order to fetch models
            Please install it from https://git-lfs.github.com/
        ]=])
    endif()
endfunction()

function (fetch_models_and_validation_set)
    foreach(loop_var ${MODELS_LST_TO_FETCH})
        string(REPLACE ":" ";" MODEL_CONFIG_LST ${loop_var})

        list(GET MODEL_CONFIG_LST 0 folder_name)
        list(GET MODEL_CONFIG_LST 1 git_url)
        list(GET MODEL_CONFIG_LST 2 repo_name)
        list(GET MODEL_CONFIG_LST 3 branch_name)

        string(FIND ${folder_name} "model" IS_MODEL)
        if(${folder_name} MATCHES "model*")
            set(FOLDER_NAME "/models/src")
        endif()
        add_lfs_repo(
            "${folder_name}"
            ${TEMP}${FOLDER_NAME}/${folder_name}
            "${git_url}:${repo_name}"
            "${branch_name}")
    endforeach(loop_var)
endfunction()
