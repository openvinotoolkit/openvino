# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gna_unit_tests)

# RPATH is always enabled for unit tests
set(CMAKE_SKIP_RPATH OFF)

# because unit tests use plugins object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

# TODO: fix CVS-71010 and remove BUILD_SHARED_LIBS
if(NOT BUILD_SHARED_LIBS)
    set(exclude_path EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/(gna_api_stub|gna_wait_test|gna_export_import_test).cpp")
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
