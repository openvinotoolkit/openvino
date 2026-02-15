# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0012 NEW)

foreach(var ONECORE_API_VALIDATOR ONECORE_API_VALIDATOR_TARGET
            ONECORE_API_VALIDATOR_APIS ONECORE_API_VALIDATOR_EXCLUSION
            ONECORE_API_VALIDATOR_OUTPUT CMAKE_TOOLCHAIN_FILE)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

# create command

if(NOT EXISTS "${ONECORE_API_VALIDATOR_APIS}")
    message(FATAL_ERROR "${ONECORE_API_VALIDATOR_APIS} does not exist")
endif()

set(command "${ONECORE_API_VALIDATOR}"
        -SupportedApiXmlFiles:${ONECORE_API_VALIDATOR_APIS}
        -DriverPackagePath:${ONECORE_API_VALIDATOR_TARGET})
if(EXISTS "${ONECORE_API_VALIDATOR_EXCLUSION}")
    list(APPEND command
        -BinaryExclusionListXmlFile:${ONECORE_API_VALIDATOR_EXCLUSION}
        -StrictCompliance:TRUE)
    set(ONECORE_HAS_BINARY_EXCLUSION ON)
endif()

# execute

execute_process(COMMAND ${command}
                OUTPUT_VARIABLE output_message
                ERROR_VARIABLE error_message
                RESULT_VARIABLE exit_code
                OUTPUT_STRIP_TRAILING_WHITESPACE)

file(WRITE "${ONECORE_API_VALIDATOR_OUTPUT}" "CMAKE COMMAND: ${command}\n\n\n${output_message}\n\n\n${error_message}")

# post-process output

get_filename_component(name "${ONECORE_API_VALIDATOR_TARGET}" NAME)

if(NOT ONECORE_HAS_BINARY_EXCLUSION)
    if(CMAKE_TOOLCHAIN_FILE MATCHES "onecoreuap.toolchain.cmake$")
        # empty since we compile with static MSVC runtime
    else()
        set(exclusion_dlls "msvcp140.dll" "vcruntime140.dll")
    endif()

    # remove exclusions from error_message

    foreach(dll IN LISTS exclusion_dlls)
        string(REGEX REPLACE
                "ApiValidation: Error: ${name} has unsupported API call to \"${dll}![^\"]+\"\n"
                "" error_message "${error_message}")
    endforeach()

    # throw error if error_message still contains any errors

    if(error_message)
        message(FATAL_ERROR "${error_message}")
    endif()
endif()

# write output

if(ONECORE_HAS_BINARY_EXCLUSION AND NOT exit_code EQUAL 0)
    message(FATAL_ERROR "${error_message}")
endif()

message("ApiValidator: ${name} has passed the OneCore compliance")
