// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/topk.hpp"

#include <algorithm>

#include "openvino/runtime/properties.hpp"

namespace {
using ov::test::TopK11LayerTest;
using ov::test::TopKLayerTest;

class TopK11LayerTestBF16GPU : public TopK11LayerTest {
protected:
    void SetUp() override {
        TopK11LayerTest::SetUp();
        const auto capabilities = core->get_property(targetDevice, ov::device::capabilities);
        if (std::find(capabilities.begin(), capabilities.end(), ov::device::capability::BF16) == capabilities.end()) {
            GTEST_SKIP() << "Device does not support BF16";
        }
        configuration.insert(ov::hint::inference_precision(ov::element::bf16));
    }

    void validate() override {
        TopK11LayerTest::validate();
        const auto runtime_model = compiledModel.get_runtime_model();
        const auto& ops = runtime_model->get_ordered_ops();
        const auto topk = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& op) {
            return op->get_rt_info().at("layerType").as<std::string>() == "ArgMax";
        });
        ASSERT_NE(topk, ops.end());
        EXPECT_EQ((*topk)->get_rt_info().at("runtimePrecision").as<std::string>(), "bf16");
    }
};

TEST_P(TopK11LayerTestBF16GPU, Inference) {
    run();
}

std::vector<ov::Shape> shapes = {{10, 10, 10}};

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<int64_t> axes = {
    0,
    1,
    2,
};

const std::vector<int64_t> keep = {
    1,
    5,
    10,
};

const std::vector<ov::op::v1::TopK::Mode> modes = {
    ov::op::v1::TopK::Mode::MIN,
    ov::op::v1::TopK::Mode::MAX,
};

const std::vector<ov::op::v1::TopK::SortType> sortTypes = {
    ov::op::v1::TopK::SortType::SORT_INDICES,
    ov::op::v1::TopK::SortType::SORT_VALUES,
};

const std::vector<bool> stable = {
    false,
    true,
};

INSTANTIATE_TEST_SUITE_P(smoke_TopK11,
                         TopK11LayerTest,
                         ::testing::Combine(::testing::ValuesIn(keep),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(stable),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         TopK11LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK11_BF16,
                         TopK11LayerTestBF16GPU,
                         ::testing::Combine(::testing::ValuesIn(keep),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(stable),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         TopK11LayerTest::getTestCaseName);

}  // namespace
