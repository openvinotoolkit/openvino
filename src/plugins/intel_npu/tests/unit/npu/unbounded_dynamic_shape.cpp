// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

// Validates the early guard in Plugin::compile_model() that rejects models
// with unbounded dynamic dimensions (upper bound = INT64_MAX) on NPU.

TEST(NPUUnboundedDynamicShape, RejectsUnboundedDynamicInput) {
    ov::Core core;

    auto devices = core.get_available_devices();
    bool npu_available = std::find(devices.begin(), devices.end(), "NPU") != devices.end();
    if (!npu_available) {
        GTEST_SKIP() << "NPU device not available";
    }

    // Dimension::dynamic() creates a dimension with range [0, INT64_MAX]
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 64});
    param->set_friendly_name("test_input");
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param});

    EXPECT_THROW(
        {
            try {
                core.compile_model(model, "NPU");
            } catch (const ov::Exception& e) {
                std::string msg = e.what();
                EXPECT_NE(msg.find("unbounded dynamic dimensions"), std::string::npos)
                    << "Error message should mention 'unbounded dynamic dimensions', got: " << msg;
                EXPECT_NE(msg.find("test_input"), std::string::npos)
                    << "Error message should mention the parameter name, got: " << msg;
                EXPECT_NE(msg.find("model.reshape"), std::string::npos)
                    << "Error message should suggest model.reshape(), got: " << msg;
                throw;
            }
        },
        ov::Exception);
}

TEST(NPUUnboundedDynamicShape, AllowsBoundedDynamicInput) {
    ov::Core core;

    auto devices = core.get_available_devices();
    bool npu_available = std::find(devices.begin(), devices.end(), "NPU") != devices.end();
    if (!npu_available) {
        GTEST_SKIP() << "NPU device not available";
    }

    // Dimension(1, 512) has a finite upper bound — should NOT be rejected by the guard
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{1, ov::Dimension(1, 512), 64});
    param->set_friendly_name("test_input_bounded");
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param});

    // Should NOT throw from our guard (may fail later in the compiler for
    // other reasons, but not from the unbounded-shape check)
    try {
        core.compile_model(model, "NPU");
    } catch (const ov::Exception& e) {
        std::string msg = e.what();
        EXPECT_EQ(msg.find("unbounded dynamic dimensions"), std::string::npos)
            << "Bounded dynamic shape should not trigger the unbounded-shape guard. Got: " << msg;
    }
}
