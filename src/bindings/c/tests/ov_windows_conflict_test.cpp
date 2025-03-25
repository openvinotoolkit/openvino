// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*
 * This test only for Windows, which caused by the BOOLEAN define conflict between Windows.h and openvino.h
 * (ov_element_type_e). Sollution: 1) check Windows.h was used or not in openvino.h for windows OS 2) if YES, will
 * redefine the "BOOLEAN" from Windows.h to "WIN_BOOLEAN" and also redefine the "BOOLEAN" from openvino.h to
 * "OV_BOOLEAN"
 */

#if defined(_WIN32)
#    include <gtest/gtest.h>
#    include <windows.h>

#    include "openvino/c/openvino.h"

#    ifndef UNUSED
#        define UNUSED(x) ((void)(x))
#    endif

TEST(ov_windows_conflict_test, ov_windows_boolean_conflict) {
    ov_element_type_e element_type = OV_BOOLEAN;  // The BOOLEAN from ov_element_type_e will be replaced by OV_BOOLEAN
    UNUSED(element_type);
}
#endif
