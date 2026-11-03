// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

class ConvolutionInPlaceCrop : public ov::test::SubgraphBaseStaticTest,
                              public ::testing::WithParamInterface<std::string> {
protected:
    std::string expected_impl_type;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& precision = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                             ov::Shape{1, 256, 20, 20});
        auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                               ov::Shape{256, 128, 1, 1});
        std::shared_ptr<ov::Node> input_data = input;
        std::shared_ptr<ov::Node> weights_data = weights;

        if (precision == "f32") {
            configuration[ov::hint::inference_precision.name()] = ov::element::f32.get_type_name();
            input_data = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
            weights_data = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
            expected_impl_type = "convolution_gpu_bfyx_os_iyx_osv16__f32";
        } else {
            configuration[ov::hint::inference_precision.name()] = ov::element::f16.get_type_name();
            expected_impl_type = "convolution_gpu_bfyx_os_iyx_osv32__f16";
            // The osv32 kernel accumulates in f16, so loosen the tolerance to
            // account for the reduced precision over the feature dimension.
            abs_threshold = 0.05;
            rel_threshold = 0.05;
        }

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64,
                                                          ov::Shape{2},
                                                          {128, 128});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(input_data,
                                                                 axis,
                                                                 split_lengths);
        auto convolution = std::make_shared<ov::op::v1::Convolution>(split->output(1),
                                                                     weights_data,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1});
        convolution->set_friendly_name("convolution");

        auto result = std::make_shared<ov::op::v0::Result>(convolution);
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{input, weights},
                                               "ConvolutionInPlaceCrop");
    }

    void validate() override {
        ov::test::SubgraphBaseStaticTest::validate();

        const auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_NE(runtime_model, nullptr);
        for (const auto& node : runtime_model->get_ordered_ops()) {
            const auto& info = node->get_rt_info();
            const auto layer_type = info.find(ov::exec_model_info::LAYER_TYPE);
            const auto impl_type = info.find(ov::exec_model_info::IMPL_TYPE);
            if (layer_type != info.end() &&
                layer_type->second.as<std::string>() == "Convolution") {
                ASSERT_NE(impl_type, info.end());
                EXPECT_EQ(impl_type->second.as<std::string>(), expected_impl_type);
                return;
            }
        }
        FAIL() << "Convolution node was not found in the runtime model";
    }
};

TEST_P(ConvolutionInPlaceCrop, smoke_GPU_second_half_feature_crop_no_oob) {
    run();
}

INSTANTIATE_TEST_SUITE_P(ConvolutionInPlaceCropPrecision,
                         ConvolutionInPlaceCrop,
                         ::testing::Values("f32", "f16"));

}  // namespace
