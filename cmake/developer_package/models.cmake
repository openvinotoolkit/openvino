# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(MODELS_LST "")
set(MODELS_LST_TO_FETCH "")

function (add_models_repo add_to_fetcher model_name)
    message(WARNING "DEPRECATED: 'add_models_repo' must not be used")

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
    message(WARNING "DEPRECATED: 'add_lfs_repo' must not be used")

    if(TARGET ${name})
        return()
    endif()

    include(ExternalProject)
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

    find_package(Git REQUIRED)

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
    message(WARNING "DEPRECATED: 'fetch_models_and_validation_set' must not be used")

    foreach(loop_var ${MODELS_LST_TO_FETCH})
        string(REPLACE ":" ";" MODEL_CONFIG_LST ${loop_var})

        list(GET MODEL_CONFIG_LST 0 folder_name)
        list(GET MODEL_CONFIG_LST 1 git_url)
        list(GET MODEL_CONFIG_LST 2 repo_name)
        list(GET MODEL_CONFIG_LST 3 branch_name)

        add_lfs_repo(
            "${folder_name}"
            "${TEMP}/models"
            "${git_url}:${repo_name}"
            "${branch_name}")
    endforeach(loop_var)
endfunction()
