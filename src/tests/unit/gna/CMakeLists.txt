# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gnaUnitTests)

# TODO: fix CVS-71010 and remove BUILD_SHARED_LIBS
if(NOT BUILD_SHARED_LIBS)
    set(exclude_path EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/(gna_api_stub|gna_wait_test).cpp")
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ${exclude_path}
        LINK_LIBRARIES
            PRIVATE
                ngraphFunctions
            gmock
            commonTestUtils_s
            openvino_intel_gna_plugin_test_static
            engines_test_util
        ADD_CPPLINT
        LABELS
            GNA
)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set_source_files_properties(gna_model_serial_test.cpp
                                PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
endif()
