// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/gna_limitations.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "memory"
#include "ngraph/function.hpp"
#include "ngraph/opsets/opset11.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/op/or.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "ngraph/rt_info.hpp"
#include "openvino/opsets/opset11.hpp"
#include "transformations/aszp_decomposition.hpp"
#include "transformations/init_node_info.hpp"
#include "tuple"

namespace testing {
namespace {

typedef std::tuple<ov::Shape,           // Input shape
                   ov::Shape,           // Convolution filter shape
                   ov::Strides,         // Convolution stride
                   ov::CoordinateDiff,  // Convolution pads begin
                   ov::CoordinateDiff,  // Convolution pads end
                   ov::Strides,         // Convolution dilation
                   ov::op::PadType>     // Padding type
    asymmetricToSymmetricConvParams;

std::shared_ptr<ov::opset11::Result> create_function(const ov::Output<ov::Node>& input_node,
                                                     const ov::Shape& filters_shape,
                                                     const ov::Strides& conv_stride,
                                                     const ov::CoordinateDiff& pads_begin,
                                                     const ov::CoordinateDiff& pads_end,
                                                     const ov::Strides& conv_dilation,
                                                     const ov::op::PadType& pad_type) {
    auto transpose_in = std::make_shared<ov::opset11::Transpose>(
        input_node,
        std::make_shared<ov::opset11::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2}));

    std::shared_ptr<ov::Node> filters = std::make_shared<ov::opset11::Constant>(
        ov::element::i64,
        ov::Shape{4, input_node.get_shape()[3], filters_shape[0], filters_shape[1]});

    auto conv = std::make_shared<ov::opset11::Convolution>(transpose_in->output(0),
                                                           filters,
                                                           conv_stride,
                                                           pads_begin,
                                                           pads_end,
                                                           conv_dilation,
                                                           pad_type);

    ov::Output<ov::Node> transpose_out = std::make_shared<ov::opset11::Transpose>(
        conv->output(0),
        std::make_shared<ov::opset11::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1}));

    return std::make_shared<ov::opset11::Result>(transpose_out);
}

std::shared_ptr<ngraph::Function> get_initial_function(const ov::Shape& input_shape,
                                                       const ov::Shape& filters_shape,
                                                       const ov::Strides& conv_stride,
                                                       const ov::CoordinateDiff& pads_begin,
                                                       const ov::CoordinateDiff& pads_end,
                                                       const ov::Strides& conv_dilation,
                                                       const ov::op::PadType& pad_type) {
    auto input_params = std::make_shared<ov::opset11::Parameter>(ov::element::i64, input_shape);
    auto result =
        create_function(input_params, filters_shape, conv_stride, pads_begin, pads_end, conv_dilation, pad_type);
    return std::make_shared<ngraph::Function>(ov::ResultVector{result}, ov::ParameterVector{input_params});
}

//===============================================================================================================================================

class asymmetricToSymmetricPaddingTestFixture : public CommonTestUtils::TestsCommon,
                                                public ::testing::WithParamInterface<asymmetricToSymmetricConvParams> {
public:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> get_reference(const ov::Shape& input_shape,
                                                    const ov::Shape& filters_shape,
                                                    const ov::Strides& conv_stride,
                                                    const ov::CoordinateDiff& pads_begin,
                                                    const ov::CoordinateDiff& pads_end,
                                                    const ov::Strides& conv_dilation,
                                                    const ov::op::PadType& pad_type);

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void asymmetricToSymmetricPaddingTestFixture::SetUp() {
    asymmetricToSymmetricConvParams params;
    ov::Shape input_shape;
    ov::Shape filters_shape;
    ov::Strides conv_stride;
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    ov::Strides conv_dilation;
    ov::op::PadType pad_type;
    params = this->GetParam();
    std::tie(input_shape, filters_shape, conv_stride, pads_begin, pads_end, conv_dilation, pad_type) = params;

    function =
        get_initial_function(input_shape, filters_shape, conv_stride, pads_begin, pads_end, conv_dilation, pad_type);
    reference_function =
        get_reference(input_shape, filters_shape, conv_stride, pads_begin, pads_end, conv_dilation, pad_type);
}

static std::tuple<int64_t, int64_t, int64_t> extract_height_padding(ov::CoordinateDiff pads_begin,
                                                                    ov::CoordinateDiff pads_end) {
    auto height_begin = pads_begin[0];
    auto height_end = pads_end[0];
    return std::make_tuple(height_begin, height_end, std::abs(height_begin - height_end));
}

static std::tuple<int64_t, int64_t, int64_t> extract_width_padding(ov::CoordinateDiff pads_begin,
                                                                   ov::CoordinateDiff pads_end) {
    auto width_begin = pads_begin[1];
    auto width_end = pads_end[1];
    return std::make_tuple(width_begin, width_end, std::abs(width_begin - width_end));
}

std::shared_ptr<ov::opset11::Transpose> create_transpose(const ov::Output<ov::Node>& input) {
    return std::make_shared<ov::opset11::Transpose>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
}

std::shared_ptr<ov::opset11::Reshape> create_reshape(const ov::Output<ov::Node>& input,
                                                     uint64_t ndims,
                                                     ov::Shape shape) {
    return std::make_shared<ov::opset11::Reshape>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{ndims}, shape)->output(0),
        false);
}

std::shared_ptr<ov::opset11::Constant> create_zero_const(ov::Shape shape) {
    return ov::opset11::Constant::create(ov::element::i64, shape, std::vector<float>(shape[0] * shape[1], 0.0f));
}

std::shared_ptr<ov::op::v0::Concat> concatenate_zeros(uint64_t pad_begin,
                                                      uint64_t pad_end,
                                                      std::shared_ptr<ov::Node> padding_const,
                                                      std::shared_ptr<ov::Node> input_node) {
    ov::OutputVector concat_vector;
    if (pad_begin > pad_end) {
        concat_vector.push_back(padding_const->output(0));
        concat_vector.push_back(input_node->output(0));
    } else {
        concat_vector.push_back(input_node->output(0));
        concat_vector.push_back(padding_const->output(0));
    }
    return std::make_shared<ov::opset11::Concat>(concat_vector, 1);
}

std::shared_ptr<ov::opset11::Transpose> get_transpose_before(std::shared_ptr<ov::Node> conv) {
    const ov::Output<ov::Node>& parent = conv->input_value(0);

    auto transpose_before = std::dynamic_pointer_cast<ov::opset11::Transpose>(parent.get_node()->shared_from_this());
    if (nullptr == transpose_before) {
        return nullptr;
    }

    auto convolution_children = conv->output(0).get_target_inputs();
    auto convolution_bias =
        std::dynamic_pointer_cast<ov::opset11::Add>(convolution_children.begin()->get_node()->shared_from_this());

    std::shared_ptr<ov::opset11::Transpose> transpose_after;
    if (nullptr != convolution_bias) {
        auto add_children = convolution_bias->output(0).get_target_inputs();
        if (add_children.size() != 1)
            return nullptr;
        transpose_after =
            std::dynamic_pointer_cast<ov::opset11::Transpose>(add_children.begin()->get_node()->shared_from_this());
    } else {
        transpose_after = std::dynamic_pointer_cast<ov::opset11::Transpose>(
            convolution_children.begin()->get_node()->shared_from_this());
    }

    if (transpose_after == nullptr) {
        return nullptr;
    }
    return transpose_before;
}

static std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> get_input_dimensions(ov::Shape input_shape) {
    uint64_t N = input_shape[0];
    uint64_t H = input_shape[1];
    uint64_t W = input_shape[2];
    uint64_t C = input_shape[3];
    return std::make_tuple(N, H, W, C);
}

ov::Output<ov::Node> decompose_height(ov::Output<ov::Node> input,
                                      ov::CoordinateDiff pads_begin,
                                      ov::CoordinateDiff pads_end,
                                      ov::Shape conv_input_shape) {
    uint64_t height_begin, height_end, height_padding, width_padding;
    std::tie(height_begin, height_end, height_padding) = extract_height_padding(pads_begin, pads_end);
    width_padding = std::abs(pads_end[1] - pads_begin[1]);
    uint64_t N, C, H, W;
    std::tie(N, H, W, C) = get_input_dimensions(conv_input_shape);

    if (0 == height_padding) {
        return input;
    }

    auto new_reshape = create_reshape(input, 2, ov::Shape{H, W * C});
    auto new_transpose = create_transpose(new_reshape->output(0));
    auto padding_const = create_zero_const(ov::Shape{W * C, height_padding});
    auto new_concat = concatenate_zeros(height_begin, height_end, padding_const, new_transpose);
    auto new_untranspose = create_transpose(new_concat->output(0));

    if (0 == width_padding) {
        return create_reshape(new_untranspose->output(0), 4, ov::Shape{N, H + height_padding, W, C})->output(0);
    }
    return (new_untranspose->output(0));
}

ov::Output<ov::Node> decompose_width(ov::Output<ov::Node> input,
                                     ov::CoordinateDiff pads_begin,
                                     ov::CoordinateDiff pads_end,
                                     ov::Shape conv_input_shape) {
    uint64_t width_begin, width_end, width_padding, height_padding;
    std::tie(width_begin, width_end, width_padding) = extract_width_padding(pads_begin, pads_end);
    height_padding = std::abs(pads_end[0] - pads_begin[0]);
    uint64_t N, H, W, C;
    std::tie(N, H, W, C) = get_input_dimensions(conv_input_shape);
    if (0 == width_padding) {
        return input;
    }

    auto new_reshape = create_reshape(input, 2, ov::Shape{(H + height_padding) * W, C});
    auto new_transpose = create_transpose(new_reshape->output(0));
    auto new_reshape2 = create_reshape(new_transpose->output(0), 2, ov::Shape{C * (H + height_padding), W});
    auto padding_const = create_zero_const(ov::Shape{C * (H + height_padding), width_padding});
    auto new_concat = concatenate_zeros(width_begin, width_end, padding_const, new_reshape2);
    auto new_unshape2 =
        create_reshape(new_concat->output(0), 2, ov::Shape{C, (H + height_padding) * (W + width_padding)});
    auto new_untranspose = create_transpose(new_unshape2->output(0));
    auto new_unshape = create_reshape(new_untranspose->output(0), 4, {N, H + height_padding, W + width_padding, C});

    return new_unshape->output(0);
}

// Create reference function
std::shared_ptr<ngraph::Function> asymmetricToSymmetricPaddingTestFixture::get_reference(
    const ov::Shape& input_shape,
    const ov::Shape& filters_shape,
    const ov::Strides& conv_stride,
    const ov::CoordinateDiff& pads_begin,
    const ov::CoordinateDiff& pads_end,
    const ov::Strides& conv_dilation,
    const ov::op::PadType& pad_type) {
    auto input_params = std::make_shared<ov::opset11::Parameter>(ov::element::i64, input_shape);
    std::shared_ptr<ov::opset11::Result> result;

    auto h_pad = std::min(pads_begin[0], pads_end[0]);
    auto w_pad = std::min(pads_begin[1], pads_end[1]);
    auto h_pad_margin = std::abs(pads_begin[0] - pads_end[0]);
    auto w_pad_margin = std::abs(pads_begin[1] - pads_end[1]);

    if (w_pad_margin + h_pad_margin) {
        ov::Output<ov::Node> skip_input_H_const = decompose_height(input_params, pads_begin, pads_end, input_shape);
        ov::Output<ov::Node> skip_input_W_const =
            decompose_width(skip_input_H_const, pads_begin, pads_end, input_shape);
        result = create_function(skip_input_W_const,
                                 filters_shape,
                                 conv_stride,
                                 ov::CoordinateDiff{h_pad, w_pad},
                                 ov::CoordinateDiff{h_pad, w_pad},
                                 conv_dilation,
                                 pad_type);
    } else {
        result =
            create_function(input_params, filters_shape, conv_stride, pads_begin, pads_end, conv_dilation, pad_type);
    }

    return std::make_shared<ngraph::Function>(ov::ResultVector{result}, ov::ParameterVector{input_params});
}

//===============================================================================================================================================

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    // manager.register_pass<ov::pass::Serialize>("pre_pass_function.xml", "pre_pass_function.bin");
    manager.register_pass<ov::intel_gna::pass::AszpDecomposition>();
    // manager.register_pass<ov::pass::Serialize>("post_pass_function.xml", "post_pass_function.bin");
    manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(asymmetricToSymmetricPaddingTestFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(asymmetricToSymmetricPaddingTestSuite,
                         asymmetricToSymmetricPaddingTestFixture,
                         ::testing::Values(std::make_tuple(ov::Shape{1, 3, 16, 5},      // Input shape
                                                           ov::Shape{1, 2},             // Convolution filter shape
                                                           ov::Strides{1, 1},           // Convolution stride
                                                           ov::CoordinateDiff{0, 3},    // Convolution pads begin
                                                           ov::CoordinateDiff{0, 2},    // Convolution pads end
                                                           ov::Strides{1, 1},           // Convolution dilation
                                                           ov::op::PadType::EXPLICIT),  // Padding type
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{0, 2},
                                                           ov::CoordinateDiff{0, 3},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT),
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{2, 0},
                                                           ov::CoordinateDiff{3, 0},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT),
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{3, 0},
                                                           ov::CoordinateDiff{2, 0},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT),
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{2, 2},
                                                           ov::CoordinateDiff{3, 3},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT),
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{2, 3},
                                                           ov::CoordinateDiff{3, 2},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT),
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{3, 2},
                                                           ov::CoordinateDiff{2, 3},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT),
                                           std::make_tuple(ov::Shape{1, 3, 16, 5},
                                                           ov::Shape{1, 2},
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{3, 3},
                                                           ov::CoordinateDiff{2, 2},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT)));

}  // namespace
}  // namespace testing