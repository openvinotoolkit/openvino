// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The rank devices come from the device name, "TP:<devices>": the core treats
// TP as a virtual device, so the list arrives as ov::device::priorities and
// ov::device::properties(<device>, ...) reach the rank compilations.

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/container_util.hpp"
#include "tp/internal_properties.hpp"
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

// A collective call is matched across the ranks by the channel of one stream.
TEST(TPDeviceNameTest, RejectsSeveralStreams) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,GPU.1", ov::num_streams(2)),
                    ov::Exception,
                    HasSubstr("one stream per device"));
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,GPU.1", ov::num_streams(ov::streams::AUTO)),
                    ov::Exception,
                    HasSubstr("one stream per device"));
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,GPU.1", ov::device::properties("GPU.1", ov::num_streams(2))),
                    ov::Exception,
                    HasSubstr("one stream per device"));
}

TEST(TPDeviceNameTest, RejectsLinearAttentionState) {
    ov::Core core;
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
    auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
    state->output(0).set_names({"conv_state_table.0"});
    auto sum = std::make_shared<ov::op::v1::Add>(x, state);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{sum}, ov::ParameterVector{x, state});
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,GPU.1"),
                    ov::Exception,
                    HasSubstr("TP does not support models with linear-attention state yet."));
}

TEST(TPDeviceNameTest, RejectsAnUnknownStrategy) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,GPU.1", ov::AnyMap{{"TP_STRATEGY", "ROW_ONLY"}}),
                    ov::Exception,
                    HasSubstr("Invalid value: ROW_ONLY for property: TP_STRATEGY"));
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

// What the rank devices can serve is the device plugin's answer, given before
// anything is compiled.
TEST_F(TPDevicePropertiesTest, RejectsAGroupTheDevicePluginCannotServe) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP:GPU.0,GPU.0"), ov::Exception, HasSubstr("more than once"));
    OV_EXPECT_THROW(core.compile_model(model, tp_device(2), ov::tp::collective_backend("NO_SUCH_BACKEND")),
                    ov::Exception,
                    HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\""));
}

TEST_F(TPDevicePropertiesTest, RejectsADevicePluginWithoutCollectives) {
    ov::Core core;
    if (!ov::util::contains(core.get_available_devices(), "CPU"))
        GTEST_SKIP() << "no CPU plugin";
    const auto model = make_transformer_block(BlockConfig{});
    OV_EXPECT_THROW(core.compile_model(model, "TP:CPU,CPU"),
                    ov::Exception,
                    HasSubstr("the CPU plugin does not support tensor-parallel collectives"));
}

// The backend name reaches the device plugin unchanged; the compiled model
// reports its rank count.
TEST_F(TPDevicePropertiesTest, NamedBackendAndWorldSize) {
    ov::Core core;
    const auto model = make_transformer_block(BlockConfig{});
    auto compiled = core.compile_model(model,
                                       tp_device(2),
                                       ov::tp::collective_backend("L0_P2P"),
                                       ov::tp::tp_strategy(ov::tp::TPStrategy::COLUMN_ROW));
    EXPECT_EQ(compiled.get_property(ov::tp::internal::world_size.name()).as<int>(), 2);
    EXPECT_EQ(compiled.get_property(ov::num_streams), ov::streams::Num{1});
}

// A TP model has no single device: its context hands out host memory only, and
// an input in that memory gives the same result as one in ordinary memory.
TEST_F(TPDevicePropertiesTest, ContextGivesHostTensorsOnly) {
    ov::Core core;
    const BlockConfig config;
    auto compiled = core.compile_model(make_transformer_block(config), tp_device(2));
    auto context = compiled.get_context();
    EXPECT_EQ(context.get_device_name(), "TP");
    OV_EXPECT_THROW(context.create_tensor(ov::element::f32, ov::Shape{1, 4}, ov::AnyMap{}),
                    ov::Exception,
                    HasSubstr("a TP model has no single device"));

    const ov::Shape shape{1, 5, config.hidden};
    ov::Tensor plain(ov::element::f32, shape);
    auto host = context.create_host_tensor(ov::element::f32, shape);
    for (size_t i = 0; i < plain.get_size(); ++i) {
        plain.data<float>()[i] = static_cast<float>(i % 13) * 0.01f;
    }
    plain.copy_to(host);

    auto request = compiled.create_infer_request();
    request.set_input_tensor(plain);
    request.infer();
    ov::Tensor expected(request.get_output_tensor().get_element_type(), request.get_output_tensor().get_shape());
    request.get_output_tensor().copy_to(expected);

    request.set_input_tensor(host);
    request.infer();
    const auto actual = request.get_output_tensor();
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.get_byte_size()), 0);
}

}  // namespace
}  // namespace ov::tp::tests
