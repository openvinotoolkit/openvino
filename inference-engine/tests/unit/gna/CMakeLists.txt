# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gnaUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            gmock
            commonTestUtils_s
            GNAPlugin_test_static
        ADD_CPPLINT
        LABELS
            GNA
)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set_source_files_properties(gna_model_serial_test.cpp
                                PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
endif()
