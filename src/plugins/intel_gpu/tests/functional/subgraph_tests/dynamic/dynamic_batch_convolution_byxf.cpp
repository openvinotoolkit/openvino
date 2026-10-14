// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

using DynamicBatchConvolutionByxfParams = std::tuple<InputShape,           // input shape
                                                     std::vector<size_t>,  // kernel
                                                     std::vector<size_t>,  // strides
                                                     size_t>;              // output channels

// The Roll branch keeps the network-wide byxf rule off, so the convolution gets byxf from the per-node rule.
class DynamicBatchConvolutionByxf : public testing::WithParamInterface<DynamicBatchConvolutionByxfParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicBatchConvolutionByxfParams>& obj) {
        const auto& [input_shape, kernel, strides, out_channels] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shape.first}) << "_TS=";
        for (const auto& shape : input_shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "K=" << ov::test::utils::vec2str(kernel) << "_S=" << ov::test::utils::vec2str(strides) << "_O=" << out_channels;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [input_shape, kernel, strides, out_channels] = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto caps = core->get_property(targetDevice, ov::device::capabilities);
        if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
            GTEST_SKIP() << "oneDNN convolutions are only used on GPUs with the HW_MATMUL capability";
        }
        abs_threshold = 0.05;
        rel_threshold = 0.05;

        init_input_shapes({input_shape});
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        auto conv = ov::test::utils::make_convolution(param,
                                                      ov::element::f16,
                                                      kernel,
                                                      strides,
                                                      {0, 0},
                                                      {0, 0},
                                                      {1, 1},
                                                      ov::op::PadType::SAME_UPPER,
                                                      out_channels,
                                                      true);
        auto relu = std::make_shared<ov::op::v0::Relu>(conv);
        auto shift = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
        auto roll = std::make_shared<ov::op::v7::Roll>(param, shift, axes);
        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(relu), std::make_shared<ov::op::v0::Result>(roll)},
                                               ov::ParameterVector{param});
    }

    // oneDNN jit:ir supports byxf at any batch. With b_fs_yx_fsv16, batches of 16 or more fall back to the reference impl.
    void check_convolution_format() const {
        size_t conv_count = 0;
        for (const auto& op : compiledModel.get_runtime_model()->get_ordered_ops()) {
            const auto& rt_info = op->get_rt_info();
            if (rt_info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() != "Convolution") {
                continue;
            }
            ++conv_count;
            EXPECT_EQ(rt_info.at(ov::exec_model_info::OUTPUT_LAYOUTS).as<std::string>(), "byxf") << op->get_friendly_name();
        }
        EXPECT_EQ(conv_count, 1u);
    }
};

TEST_P(DynamicBatchConvolutionByxf, Inference) {
    run();
    check_convolution_format();
}

const std::vector<InputShape> input_shapes = {
    {{-1, 64, 28, 28}, {{1, 64, 28, 28}, {16, 64, 28, 28}, {25, 64, 28, 28}, {8, 64, 28, 28}}},
    {{-1, 128, 14, 14}, {{25, 128, 14, 14}, {1, 128, 14, 14}, {17, 128, 14, 14}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchConvolutionByxf,
                         DynamicBatchConvolutionByxf,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(std::vector<size_t>{3, 3}, std::vector<size_t>{1, 1}),
                                            ::testing::Values(std::vector<size_t>{1, 1}, std::vector<size_t>{2, 2}),
                                            ::testing::Values(size_t{128})),
                         DynamicBatchConvolutionByxf::getTestCaseName);
}  // namespace
