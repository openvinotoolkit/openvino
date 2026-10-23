// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// Conv -> GroupNormalization + Swish -> Conv with a dynamic batch: GroupNormalization takes byxf, so the network-wide
// byxf rule keeps both oneDNN convolutions in byxf without reorders around GroupNormalization.
class DynamicBatchConvGroupNormByxf : public testing::WithParamInterface<InputShape>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InputShape>& obj) {
        const auto& input_shape = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shape.first}) << "_TS=";
        for (const auto& shape : input_shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto caps = core->get_property(targetDevice, ov::device::capabilities);
        if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
            GTEST_SKIP() << "oneDNN convolutions are only used on GPUs with the HW_MATMUL capability";
        }
        // f32 model run in f16 on GPU: the template reference stays in f32.
        configuration.insert(ov::hint::inference_precision(ov::element::f16));
        abs_threshold = 0.05;
        rel_threshold = 0.05;

        init_input_shapes({GetParam()});
        const auto type = ov::element::f32;
        const auto channels = static_cast<size_t>(inputDynamicShapes[0][1].get_length());
        auto make_conv = [&](const ov::Output<ov::Node>& input, int seed) {
            const auto weights = ov::test::utils::generate_float_numbers(channels * channels * 9, -0.05f, 0.05f, seed);
            return ov::test::utils::make_convolution(input, type, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT, channels, false, weights);
        };

        auto param = std::make_shared<ov::op::v0::Parameter>(type, inputDynamicShapes[0]);
        auto conv1 = make_conv(param, 1);
        auto scale = ov::test::utils::make_constant(type, ov::Shape{channels}, ov::test::utils::generate_float_numbers(channels, 0.5f, 1.5f, 2));
        auto bias = ov::test::utils::make_constant(type, ov::Shape{channels}, ov::test::utils::generate_float_numbers(channels, -0.5f, 0.5f, 3));
        auto group_norm = std::make_shared<ov::op::v12::GroupNormalization>(conv1, scale, bias, 32, 1e-5);
        auto swish = std::make_shared<ov::op::v4::Swish>(group_norm);
        auto conv2 = make_conv(swish, 4);
        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(conv2)}, ov::ParameterVector{param});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& param = function->get_parameters().front();
        const ov::test::utils::InputGenerateData data(-1, 2, 32);
        inputs.insert({param, ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[0], data)});
    }

    // With b_fs_yx_fsv16, oneDNN convolutions fall back to the reference impl once the runtime batch reaches 16.
    void check_layouts() const {
        size_t conv_count = 0;
        size_t group_norm_count = 0;
        for (const auto& op : compiledModel.get_runtime_model()->get_ordered_ops()) {
            const auto& type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (type == "Convolution") {
                ++conv_count;
            } else if (type == "group_normalization") {
                ++group_norm_count;
            } else {
                continue;
            }
            EXPECT_EQ(op->get_rt_info().at(ov::exec_model_info::OUTPUT_LAYOUTS).as<std::string>(), "byxf") << op->get_friendly_name();

            const auto producer = op->get_input_node_shared_ptr(0);
            const auto& producer_type = producer->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            const bool is_input_reorder = producer_type == "Reorder" && ov::is_type<ov::op::v0::Parameter>(producer->get_input_node_shared_ptr(0));
            EXPECT_TRUE(producer_type != "Reorder" || is_input_reorder) << "Reorder before " << op->get_friendly_name();
        }
        EXPECT_EQ(conv_count, 2u);
        EXPECT_EQ(group_norm_count, 1u);
    }
};

TEST_P(DynamicBatchConvGroupNormByxf, Inference) {
    run();
    check_layouts();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchConvGroupNormByxf,
                         DynamicBatchConvGroupNormByxf,
                         ::testing::Values(InputShape{{-1, 320, 32, 32}, {{1, 320, 32, 32}, {16, 320, 32, 32}, {25, 320, 32, 32}}}),
                         DynamicBatchConvGroupNormByxf::getTestCaseName);
}  // namespace
