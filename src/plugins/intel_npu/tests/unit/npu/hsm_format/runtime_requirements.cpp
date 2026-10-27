// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

namespace {}

using namespace intel_npu;
using testing::_;

using RuntimeRequirementsTest = testing::Test;

TEST_F(RuntimeRequirementsTest, CtorAndBasicGetters) {
    // The empty case first
    RuntimeRequirements runtime_requirements({}, CRE(), {});
    ASSERT_TRUE(runtime_requirements.get_sections_requirements().empty());
    ASSERT_TRUE(runtime_requirements.get_cre().empty());
    ASSERT_TRUE(runtime_requirements.get_section_id_to_type_mapping().empty());
}
