// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/deconv_1d_decomposition.hpp"

#include <gtest/gtest.h>
#include <sstream>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

// extra ops used to build explicit expected graph
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {
// Small helpers to keep expected-graph construction concise
inline std::shared_ptr<ov::op::v0::Constant> c0_i32() { return ov::op::v0::Constant::create(ov::element::i32, {}, {0}); }
inline std::shared_ptr<ov::op::v0::Constant> c1_i32() { return ov::op::v0::Constant::create(ov::element::i32, {}, {1}); }
inline std::shared_ptr<ov::op::v0::Constant> c2_i32() { return ov::op::v0::Constant::create(ov::element::i32, {}, {2}); }
inline std::shared_ptr<ov::op::v0::Constant> c_1_i32() { return ov::op::v0::Constant::create(ov::element::i32, {}, {-1}); }
inline std::shared_ptr<ov::op::v0::Constant> c1_1d() { return ov::op::v0::Constant::create(ov::element::i32, {1}, {1}); }
inline std::shared_ptr<ov::op::v0::Constant> c0_1d() { return ov::op::v0::Constant::create(ov::element::i32, {1}, {0}); }
inline std::shared_ptr<ov::op::v0::Constant> c3_1d() { return ov::op::v0::Constant::create(ov::element::i32, {1}, {3}); }
inline std::shared_ptr<ov::op::v0::Constant> c_1_1d() { return ov::op::v0::Constant::create(ov::element::i32, {1}, {-1}); }
inline ov::Output<ov::Node> make1d(const ov::Output<ov::Node>& scalar) {
    return ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(scalar, c0_i32());
}
inline ov::Output<ov::Node> concat0(const std::vector<ov::Output<ov::Node>>& dims) {
    if (dims.size() == 1) return dims[0];
    return ov::op::util::make_try_fold<ov::op::v0::Concat>(dims, 0);
}
inline ov::Output<ov::Node> shape_of_i32(const ov::Output<ov::Node>& node) {
    return ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(node, ov::element::i32);
}
inline ov::Output<ov::Node> gather_dim(const ov::Output<ov::Node>& shape, int idx) {
    auto idx_c = ov::op::v0::Constant::create(ov::element::i32, {}, {idx});
    return ov::op::util::make_try_fold<ov::op::v8::Gather>(shape, idx_c, c0_i32());
}
}  // namespace

struct DeconvTestParams {
    ov::PartialShape input_shape;
    ov::Shape weights_shape;
    size_t stride = 1;
    ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    bool expect_transformation = true;
    bool with_output_shape = false;  // Test 3-input variant
};

class DeconvDecomposition1DTest : public TransformationTestsF, public ::testing::WithParamInterface<DeconvTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DeconvTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "IS=" << p.input_shape
               << "_WS=" << p.weights_shape
               << "_S=" << p.stride
               << "_PT=" << p.pad_type
               << "_WithOS=" << p.with_output_shape
               << "_ExpectTransf=" << p.expect_transformation;
        return result.str();
    }
    static std::shared_ptr<ov::Model> buildExpectedModel(const DeconvTestParams& p) {
        // Only 1D (rank-3) cases are expected to transform
        if (p.input_shape.size() != 3)
            return nullptr;

        auto input_e = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p.input_shape);
        auto weights_e = ov::opset1::Constant::create(ov::element::f32, p.weights_shape, {1});

        auto zero_scalar = c0_i32();
        auto one_scalar = c1_i32();
        auto two_scalar = c2_i32();
        auto minus_one = c_1_i32();

        auto one_const = c1_1d();
        auto zero_1d = c0_1d();
        auto three_1d = c3_1d();

        // Reshape input [N, C, L] -> [N, C, L, 1]
        auto shape_of_input = shape_of_i32(input_e);
        auto N_node = make1d(gather_dim(shape_of_input, 0));
        auto C_in_node = make1d(gather_dim(shape_of_input, 1));
        auto L_node = make1d(gather_dim(shape_of_input, 2));
        auto input_4d_pattern = concat0({N_node, C_in_node, L_node, one_const});
        ov::Output<ov::Node> result = std::make_shared<ov::op::v1::Reshape>(input_e, input_4d_pattern, false);

        // Optional upsampling for stride > 1
        if (p.stride > 1) {
            auto stride_const = ov::op::v0::Constant::create(ov::element::i32, {}, {static_cast<int32_t>(p.stride)});
            auto stride_const_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {static_cast<int32_t>(p.stride)});

            auto L_minus_1 = ov::op::util::make_try_fold<ov::op::v1::Add>(L_node, c_1_1d());
            auto L_minus_1_times_stride = ov::op::util::make_try_fold<ov::op::v1::Multiply>(L_minus_1, stride_const_1d);
            auto upsampled_L_node = ov::op::util::make_try_fold<ov::op::v1::Add>(L_minus_1_times_stride, one_const);

            auto zero_shape = concat0({N_node, C_in_node, upsampled_L_node, one_const});
            auto zero_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
            auto zero_tensor = std::make_shared<ov::op::v3::Broadcast>(zero_value, zero_shape);

            auto L_scalar = ov::op::util::make_try_fold<ov::op::v0::Squeeze>(L_node, c0_i32());
            auto range = ov::op::util::make_try_fold<ov::op::v4::Range>(c0_i32(), L_scalar, c1_i32(), ov::element::i32);
            auto indices = ov::op::util::make_try_fold<ov::op::v1::Multiply>(range, stride_const);

            auto N_times_C_in = ov::op::util::make_try_fold<ov::op::v1::Multiply>(N_node, C_in_node);
            auto input_3d_shape = concat0({N_times_C_in, L_node, one_const});
            auto input_3d = std::make_shared<ov::op::v1::Reshape>(result, input_3d_shape, true);

            auto zero_3d_shape = concat0({N_times_C_in, upsampled_L_node, one_const});
            auto zero_3d = std::make_shared<ov::op::v1::Reshape>(zero_tensor, zero_3d_shape, true);
            auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(zero_3d, indices, input_3d, one_scalar);

            auto upsampled_4d_shape = concat0({N_node, C_in_node, upsampled_L_node, one_const});
            result = std::make_shared<ov::op::v1::Reshape>(scatter, upsampled_4d_shape, false);
        }

        // Padding computation mirrors the pass for tested cases
        bool is_auto_pad = (p.pad_type != ov::op::PadType::EXPLICIT && p.pad_type != ov::op::PadType::NOTSET);
        bool can_calculate_padding = p.input_shape.is_static();

        ov::Output<ov::Node> pad_left_dyn, pad_right_dyn;
        auto shape_of_weights = shape_of_i32(weights_e);
        auto K_dyn = ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_of_weights, c2_i32(), c0_i32());

        if (is_auto_pad && !can_calculate_padding) {
            auto K_minus_1 = ov::op::util::make_try_fold<ov::op::v1::Add>(K_dyn, minus_one);
            auto two_const = c2_i32();
            auto pad_val = ov::op::util::make_try_fold<ov::op::v1::Divide>(K_minus_1, two_const, true);

            if (p.pad_type == ov::op::PadType::SAME_UPPER) {
                pad_left_dyn = pad_val;  // floor((K-1)/2)
                pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_left_dyn);
            } else if (p.pad_type == ov::op::PadType::SAME_LOWER) {
                pad_left_dyn = ov::op::util::make_try_fold<ov::op::v1::Divide>(K_dyn, two_const, true);
                pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_left_dyn);
            } else {
                pad_left_dyn = zero_scalar;
                pad_right_dyn = zero_scalar;
            }
        } else {
            int32_t pad_begin = 0;
            int32_t pad_end = 0;
            int32_t out_pad = 0;
            auto K_minus_1 = ov::op::util::make_try_fold<ov::op::v1::Add>(K_dyn, minus_one);
            auto pad_begin_const = ov::op::v0::Constant::create(ov::element::i32, {}, {pad_begin});
            auto pad_end_const = ov::op::v0::Constant::create(ov::element::i32, {}, {pad_end});
            auto out_pad_const = ov::op::v0::Constant::create(ov::element::i32, {}, {out_pad});

            pad_left_dyn = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_begin_const);
            auto K_minus_1_minus_pad_end = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_end_const);
            pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Add>(K_minus_1_minus_pad_end, out_pad_const);
        }

        pad_left_dyn = ov::op::util::make_try_fold<ov::op::v1::Maximum>(pad_left_dyn, zero_scalar);
        pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Maximum>(pad_right_dyn, zero_scalar);

        auto pad_left_1d = make1d(pad_left_dyn);
        auto pad_right_1d = make1d(pad_right_dyn);
        auto pads_begin_dyn = concat0({zero_1d, zero_1d, pad_left_1d, zero_1d});
        auto pads_end_dyn = concat0({zero_1d, zero_1d, pad_right_1d, zero_1d});

        auto pad_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
        result = std::make_shared<ov::op::v1::Pad>(result, pads_begin_dyn, pads_end_dyn, pad_value, ov::op::PadMode::CONSTANT);

        // Weights: reshape to 4D and transpose
        auto weights_shape_1d = shape_of_i32(weights_e);
        auto weights_4d_shape = concat0({weights_shape_1d, c1_1d()});
        auto weights_4d = std::make_shared<ov::op::v1::Reshape>(weights_e, weights_4d_shape, false);
        std::vector<int32_t> transpose_order = {1, 0, 2, 3};
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {4}, transpose_order);
        auto weights_transposed = std::make_shared<ov::op::v1::Transpose>(weights_4d, transpose_const);

        // 2D Convolution
        ov::Strides conv_strides{1, 1};
        ov::CoordinateDiff conv_pads_begin{0, 0};
        ov::CoordinateDiff conv_pads_end{0, 0};
        ov::Strides conv_dilations{1, 1};
        auto conv = std::make_shared<ov::op::v1::Convolution>(result, weights_transposed,
                                                              conv_strides, conv_pads_begin, conv_pads_end,
                                                              conv_dilations, ov::op::PadType::EXPLICIT);

        // Reshape back to 3D using first 3 dims of conv shape
        auto conv_shape = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(conv, ov::element::i32);
        auto output_3d_shape = ov::op::util::make_try_fold<ov::op::v8::Slice>(conv_shape, zero_1d, three_1d, one_const);
        auto output_3d = std::make_shared<ov::op::v1::Reshape>(conv, output_3d_shape, false);

        return std::make_shared<ov::Model>(ov::NodeVector{output_3d}, ov::ParameterVector{input_e});
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        const auto rank = p.input_shape.size();

        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p.input_shape);
        auto weights = ov::opset1::Constant::create(ov::element::f32, p.weights_shape, {1});

        // Create strides/pads based on dimensionality (3D vs 4D)
        ov::Strides strides = (rank == 3) ? ov::Strides{p.stride} : ov::Strides{1, 1};
        ov::CoordinateDiff pads = (rank == 3) ? ov::CoordinateDiff{0} : ov::CoordinateDiff{0, 0};
        ov::Strides dilations = (rank == 3) ? ov::Strides{1} : ov::Strides{1, 1};

        std::shared_ptr<ov::Node> deconv;
        if (p.with_output_shape) {
            // Test 3-input variant with explicit output shape (only spatial dimensions)
            ov::Shape output_shape;
            if (rank == 3) {
                // For 3D case, only specify the spatial dimension (L)
                auto in_size = p.input_shape[2].is_static() ? p.input_shape[2].get_length() : 100;
                auto kernel_size = p.weights_shape[2];
                auto out_size = p.stride * (in_size - 1) + kernel_size;
                output_shape = {static_cast<unsigned int>(out_size)};  // Only spatial dimension
            } else {
                output_shape = {100, 100};  // 4D case - two spatial dimensions
            }
            auto output_shape_const =
                ov::opset1::Constant::create(ov::element::i64, {output_shape.size()}, output_shape);
            deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                           weights,
                                                                           output_shape_const,
                                                                           strides,
                                                                           pads,
                                                                           pads,
                                                                           dilations,
                                                                           p.pad_type);
        } else {
            // Test 2-input variant
            deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                           weights,
                                                                           strides,
                                                                           pads,
                                                                           pads,
                                                                           dilations,
                                                                           p.pad_type);
        }

        model = std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});

        if (p.expect_transformation && rank == 3) {
            model_ref = buildExpectedModel(p);
        }

        manager.register_pass<Deconv1DDecomposition>();
    }
};

TEST_P(DeconvDecomposition1DTest, CompareFunctions) {}

INSTANTIATE_TEST_SUITE_P(DeconvDecomposition1DTests,
                         DeconvDecomposition1DTest,
                         ::testing::Values(
                             // 3D Static tests (2-input variant)
                             DeconvTestParams{ov::PartialShape{1, 128, 100},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              false},
                             // 3D Static test (3-input variant with output shape)
                             DeconvTestParams{ov::PartialShape{1, 128, 100},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              true},
                             // 3D Dynamic tests
                             DeconvTestParams{ov::PartialShape{1, 2, -1},
                                              ov::Shape{2, 2, 3},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              false},
                             DeconvTestParams{ov::PartialShape{1, 128, -1},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              false},
                             DeconvTestParams{ov::PartialShape{1, 128, ov::Dimension::dynamic()},
                                              ov::Shape{128, 64, 5},
                                              2,
                                              ov::op::PadType::SAME_UPPER,
                                              true,
                                              false},
                             // 3D Dynamic test (3-input variant)
                             DeconvTestParams{ov::PartialShape{1, 128, -1},
                                              ov::Shape{128, 64, 5},
                                              2,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              true},
                             // 4D Negative test
                             DeconvTestParams{ov::PartialShape{1, 128, 100, 100},
                                              ov::Shape{128, 64, 5, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              false,  // Should NOT transform
                                              false}),
                         DeconvDecomposition1DTest::getTestCaseName);
