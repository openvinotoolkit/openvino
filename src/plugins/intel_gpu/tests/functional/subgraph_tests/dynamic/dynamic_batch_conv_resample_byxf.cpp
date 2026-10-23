// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// Conv -> Interpolate (nearest 2x) -> Conv with a dynamic batch: the resample takes byxf, so the network-wide byxf rule
// keeps both oneDNN convolutions in byxf without reorders around the resample.
class DynamicBatchConvResampleByxf : public testing::WithParamInterface<InputShape>, virtual public ov::test::SubgraphBaseTest {
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
        using Interpolate = ov::op::v11::Interpolate;
        Interpolate::InterpolateAttrs attrs;
        attrs.mode = Interpolate::InterpolateMode::NEAREST;
        attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SCALES;
        attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
        attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
        auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 2.0f});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
        auto resample = std::make_shared<Interpolate>(conv1, scales, axes, attrs);
        auto conv2 = make_conv(resample, 2);
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
        size_t resample_count = 0;
        for (const auto& op : compiledModel.get_runtime_model()->get_ordered_ops()) {
            const auto& type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (type == "Convolution") {
                ++conv_count;
            } else if (type == "Resample") {
                ++resample_count;
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
        EXPECT_EQ(resample_count, 1u);
    }
};

TEST_P(DynamicBatchConvResampleByxf, Inference) {
    run();
    check_layouts();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchConvResampleByxf,
                         DynamicBatchConvResampleByxf,
                         ::testing::Values(InputShape{{-1, 128, 16, 16}, {{1, 128, 16, 16}, {16, 128, 16, 16}, {25, 128, 16, 16}}}),
                         DynamicBatchConvResampleByxf::getTestCaseName);
}  // namespace
