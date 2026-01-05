// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include <random>

#include "common/functions.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/serialize.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "vcl_serializer.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace {

// As of writing this, the "all weights copy" solution serializes the model below using ~500KB. The "no weights copy"
// uses ~100KB. If sizes change significantly, then investigation may be required.
constexpr size_t SERIALIZED_MODEL_THRESHOLD_ALL_WEIGHTS_COPY = 300000;
constexpr size_t SERIALIZED_MODEL_THRESHOLD_NO_WEIGHTS_COPY = 200000;

std::shared_ptr<ov::Model> createModelWithLargeWeights(const bool placeOneWeightlessCacheAttribute = false) {
    auto data = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{100000});
    auto mul_constant = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.5});
    auto mul = std::make_shared<ov::opset11::Multiply>(data, mul_constant);
    auto add_constant = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {0.5});
    auto add = std::make_shared<ov::opset11::Add>(mul, add_constant);

    if (placeOneWeightlessCacheAttribute) {
        add_constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(add_constant->get_byte_size(), 0, add_constant->get_element_type());
    }

    // Just a sample model here, large iteration to make the model large
    for (int i = 0; i < 100; i++) {
        add_constant = ov::opset11::Constant::create(ov::element::f32, ov::Shape{100000}, {0.5});
        add = std::make_shared<ov::opset11::Add>(add, add_constant);
    }
    auto res = std::make_shared<ov::opset11::Result>(add);

    return std::make_shared<ov::Model>(ov::ResultVector{std::move(res)}, ov::ParameterVector{std::move(data)});
}

class TestNonDeterministicAttribute : public ov::RuntimeAttribute {
public:
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        int test_value = 0;
        visitor.on_attribute("test123", test_value);
        return true;
    }

    bool is_deterministic() const override {
        return false;
    }
};

class TestDeterministicAttribute : public ov::RuntimeAttribute {
public:
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        int test_value = 0;
        visitor.on_attribute("test234", test_value);
        return true;
    }

    bool is_deterministic() const override {
        return true;
    }
};

}  // namespace

namespace ov::test::behavior {

class DriverCompilerAdapterCustomStreamTestNPU : public ov::test::behavior::OVPluginTestBase,
                                                 public testing::WithParamInterface<CompilationParams> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

protected:
    ov::AnyMap configuration;
};

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, TestLargeModelWeightsCopy) {
    auto model = createModelWithLargeWeights();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(serializedModel =
                        ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, true));
    // If the size changes significantly, then investigation may be required
    ASSERT_TRUE(serializedModel.size > SERIALIZED_MODEL_THRESHOLD_ALL_WEIGHTS_COPY);
}

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, TestLargeModelNoWeightsCopy) {
    auto model = createModelWithLargeWeights();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(serializedModel =
                        ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false));
    // If the size changes significantly, then investigation may be required
    ASSERT_TRUE(serializedModel.size < SERIALIZED_MODEL_THRESHOLD_NO_WEIGHTS_COPY);

    ov::pass::StreamSerialize::DataHeader dataHeader;
    memcpy(&dataHeader, serializedModel.buffer.get(), sizeof(dataHeader));
    ASSERT_TRUE(dataHeader.consts_size == 0);
}

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, CheckHashPresence) {
    auto model = createModelWithLargeWeights();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(serializedModel =
                        ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false));
    ASSERT_FALSE(serializedModel.hash.has_value());

    EXPECT_NO_THROW(serializedModel =
                        ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, true));
    ASSERT_TRUE(serializedModel.hash.has_value());
}

/**
 * @brief The serialization function is able to store the WeightlessCacheAttribute using the predetermined contract
 * expected by the driver-compiler adapter.
 */
TEST_P(DriverCompilerAdapterCustomStreamTestNPU, CheckWeightlessCacheAttributePresence) {
    auto model = createModelWithLargeWeights(true);
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(
        serializedModel =
            ::intel_npu::driver_compiler_utils::serializeIR(model->clone(), dummyCompilerVersion, 11, false));
    ASSERT_FALSE(model->has_rt_info("ws_bin_offset_1"));

    EXPECT_NO_THROW(
        serializedModel =
            ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, false, true));
    // Follows the contract established with the driver-compiler adapter. Predefined prefix + a topological ID of the
    // Constant node
    ASSERT_TRUE(model->has_rt_info("ws_bin_offset_1"));
}

/**
 * @brief The custom format used by the plugin to store the WeightlessCacheAttribute changes the value of the hash.
 */
TEST_P(DriverCompilerAdapterCustomStreamTestNPU, CheckWeightlessCacheAttributeChangesHash) {
    auto model = createModelWithLargeWeights(false);
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(
        serializedModel =
            ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, true, true));
    ASSERT_TRUE(serializedModel.hash.has_value());
    const uint64_t hashNoWCA = serializedModel.hash.value();

    model = createModelWithLargeWeights(true);
    EXPECT_NO_THROW(
        serializedModel =
            ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, true, true));

    ASSERT_FALSE(hashNoWCA == serializedModel.hash.value());
}

/**
 * @brief The hash produced by the serialization function should ignore only non-deterministic runtime attributes.
 */
TEST_P(DriverCompilerAdapterCustomStreamTestNPU, CheckPluginHashIgnoresOnlyNondeterministicField) {
    auto model = createModelWithLargeWeights();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(serializedModel = ::intel_npu::driver_compiler_utils::serializeIR(model->clone(),
                                                                                      dummyCompilerVersion,
                                                                                      11,
                                                                                      false,
                                                                                      true,
                                                                                      false));
    ASSERT_TRUE(serializedModel.hash.has_value());
    const uint64_t hashNoAttribute = serializedModel.hash.value();

    model->input().get_node()->get_rt_info()[TestNonDeterministicAttribute::get_type_info_static()] =
        TestNonDeterministicAttribute();
    EXPECT_NO_THROW(serializedModel = ::intel_npu::driver_compiler_utils::serializeIR(model->clone(),
                                                                                      dummyCompilerVersion,
                                                                                      11,
                                                                                      false,
                                                                                      true,
                                                                                      false));
    ASSERT_TRUE(hashNoAttribute == serializedModel.hash.value());

    model->input().get_node()->get_rt_info()[TestDeterministicAttribute::get_type_info_static()] =
        TestDeterministicAttribute();
    EXPECT_NO_THROW(
        serializedModel =
            ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, true, false));
    ASSERT_FALSE(hashNoAttribute == serializedModel.hash.value());
}

/**
 * @brief The plugin should produce the same hash for the same model, different instances.
 */
TEST_P(DriverCompilerAdapterCustomStreamTestNPU, CheckSameModelDifferentInstancesSamePluginHash) {
    auto model = createModelWithLargeWeights();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    ::intel_npu::SerializedIR serializedModel;
    EXPECT_NO_THROW(serializedModel =
                        ::intel_npu::driver_compiler_utils::serializeIR(model,
                                                                        dummyCompilerVersion,
                                                                        10,  // Triggers an additional pass
                                                                        false,
                                                                        true,
                                                                        false));
    ASSERT_TRUE(serializedModel.hash.has_value());
    const uint64_t hashFirstInstance = serializedModel.hash.value();

    model = createModelWithLargeWeights();
    EXPECT_NO_THROW(
        serializedModel =
            ::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 10, false, true, false));
    ASSERT_TRUE(hashFirstInstance == serializedModel.hash.value());
}

const std::vector<ov::AnyMap> configs = {
    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DriverCompilerAdapterCustomStreamTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterCustomStreamTestNPU::getTestCaseName);

}  // namespace ov::test::behavior
