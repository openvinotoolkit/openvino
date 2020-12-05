# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0053 NEW)

foreach(var MANIFEST_TOOL MANIFEST_FILE MANIFEST_TARGET_FILE MANIFEST_TARGET_NAME
            MANIFEST_DEPENDENCIES MANIFEST_TOKEN MANIFEST_VERSION MANIFEST_TARGET_TYPE)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

# generate manifest file

set(MANIFEST_CONTENT
"<assembly xmlns='urn:schemas-microsoft-com:asm.v1' manifestVersion='1.0'>
  <assemblyIdentity type='win32' name='@MANIFEST_TARGET_NAME@' version='@MANIFEST_VERSION@' processorArchitecture='x86' publicKeyToken='@MANIFEST_TOKEN@' />
  <file name=\"@MANIFEST_TARGET_FILE_NAME@\" hash=\"@MANIFEST_SHA1@\" hashalg=\"SHA1\"/>@MANIFEST_DEPENDENCIES_BODY@
</assembly>")

set(dependency_template
"  <dependency>
    <dependentAssembly>
    <assemblyIdentity type='win32' name='@MANIFEST_DEPENDENCY@' version='@MANIFEST_VERSION@' processorArchitecture='x86' publicKeyToken='@MANIFEST_TOKEN@' />
    </dependentAssembly>
  </dependency>")

set(MANIFEST_DEPENDENCIES_BODY "")
string(REPLACE "." ";" MANIFEST_DEPENDENCIES "${MANIFEST_DEPENDENCIES}")

foreach(dep IN LISTS MANIFEST_DEPENDENCIES)
    set(depenency "${dependency_template}")
    set(MANIFEST_DEPENDENCY "${dep}")
    foreach(var MANIFEST_DEPENDENCY MANIFEST_VERSION MANIFEST_TOKEN)
        string(REPLACE "@${var}@" "${${var}}" depenency "${depenency}")
    endforeach()
    set(MANIFEST_DEPENDENCIES_BODY "${MANIFEST_DEPENDENCIES_BODY}${depenency}")
endforeach()

if(MANIFEST_DEPENDENCIES_BODY)
    set(MANIFEST_DEPENDENCIES_BODY "\n${MANIFEST_DEPENDENCIES_BODY}")
endif()

file(SHA1 "${MANIFEST_TARGET_FILE}" MANIFEST_SHA1)
get_filename_component(MANIFEST_TARGET_FILE_NAME ${MANIFEST_TARGET_FILE} NAME)

foreach(var MANIFEST_TARGET_NAME MANIFEST_TARGET_FILE_NAME MANIFEST_SHA1
            MANIFEST_DEPENDENCIES_BODY MANIFEST_TOKEN MANIFEST_VERSION)
    string(REPLACE "@${var}@" "${${var}}" MANIFEST_CONTENT "${MANIFEST_CONTENT}")
endforeach()

file(WRITE "${MANIFEST_FILE}" "${MANIFEST_CONTENT}")
message("Generated ${MANIFEST_FILE}")

# test manifest for correctness

execute_process(COMMAND ${MANIFEST_TOOL}
        -manifest "${MANIFEST_FILE}" -verbose -validate_manifest
    OUTPUT_VARIABLE error_message
    RESULT_VARIABLE exit_code
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR "${error_message}")
endif()

# embed manifest into a binary

if(MANIFEST_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
    set(resource_id "2")
elseif(MANIFEST_TARGET_TYPE STREQUAL "EXECUTABLE")
    set(resource_id "1")
endif()

execute_process(COMMAND ${MANIFEST_TOOL}
        -manifest "${MANIFEST_FILE}"
        "-outputresource:${MANIFEST_TARGET_FILE};#${resource_id}"
    OUTPUT_VARIABLE error_message
    RESULT_VARIABLE exit_code
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR "${error_message}")
endif()
