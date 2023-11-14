# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_unit_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4305)
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4050)
    ie_add_compiler_flags(/wd4250)
    ie_add_compiler_flags(/wd4334)
    ie_add_compiler_flags(/wd4661)
    ie_add_compiler_flags(/wd4273)
    ie_add_compiler_flags(/wd4309)
    ie_add_compiler_flags(/wd4804)
endif()
if(NOT MSVC)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-odr)
    ie_add_compiler_flags(-Wno-all)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            template_extension
        LINK_LIBRARIES
            unitTestUtils
        INCLUDES
            # for static ie_plugins.hpp
            "${CMAKE_BINARY_DIR}/src/inference/"
        LABELS
            OV
)
