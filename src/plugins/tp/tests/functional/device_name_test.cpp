// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The rank devices come from the device name, "TP:<devices>": the core treats
// TP as a virtual device, so the list arrives as ov::device::priorities and
// ov::device::properties(<device>, ...) reach the rank compilations.

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::HasSubstr;

// Rejected before any device is touched, so no GPU is needed.
TEST(TPDeviceNameTest, RequiresAtLeastTwoDevices) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP"), ov::Exception, HasSubstr("at least 2 devices, got 0"));
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0"), ov::Exception, HasSubstr("at least 2 devices, got 1"));
}

TEST(TPDeviceNameTest, RejectsMixedDeviceTypes) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,CPU"), ov::Exception, HasSubstr("one device type"));
}

TEST(TPDeviceNameTest, RejectsEmptyEntry) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,,GPU.1"), ov::Exception, HasSubstr("empty entry"));
}

class TPDevicePropertiesTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            gpu_count = ov::Core{}.get_property("GPU", ov::available_devices).size();
        } catch (const std::exception& e) {
            gpu_error = e.what();
        }
    }

    void SetUp() override {
        if (gpu_count < 2)
            GTEST_SKIP() << "need >= 2 GPUs, found " << gpu_count << gpu_error;
    }

    static size_t gpu_count;
    static std::string gpu_error;
};

size_t TPDevicePropertiesTest::gpu_count = 0;
std::string TPDevicePropertiesTest::gpu_error;

// A property for the device type reaches the ranks; the compiled model answers
// rank properties from rank 0.
TEST_F(TPDevicePropertiesTest, DeviceTypePropertiesReachRanks) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    for (const auto precision : {ov::element::f32, ov::element::f16}) {
        auto compiled = core.compile_model(model,
                                           tp_device(2),
                                           ov::device::properties("GPU", ov::hint::inference_precision(precision)));
        EXPECT_EQ(compiled.get_property(ov::hint::inference_precision), precision);
    }
}

// A property for one device applies to that rank only, over the device type's.
TEST_F(TPDevicePropertiesTest, PerDevicePropertiesOverrideDeviceType) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    auto compiled = core.compile_model(model,
                                       tp_device(2),
                                       ov::device::properties("GPU", ov::hint::inference_precision(ov::element::f16)),
                                       ov::device::properties("GPU.0", ov::hint::inference_precision(ov::element::f32)));
    EXPECT_EQ(compiled.get_property(ov::hint::inference_precision), ov::element::f32);
}

}  // namespace
}  // namespace ov::tp::tests
