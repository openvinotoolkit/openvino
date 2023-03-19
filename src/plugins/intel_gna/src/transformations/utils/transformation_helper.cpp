// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformation_helper.hpp"

#include "openvino/opsets/opset7.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include "ops/gna_convolution.hpp"
#include "ops/gna_max_pool.hpp"

using namespace ov::opset7;

namespace ov {
namespace intel_gna {
namespace pass {
namespace helper {

std::function<bool(ngraph::Output<ngraph::Node>)> consumers_and_rank(const size_t expected_count,
                                                                     const ngraph::Dimension& expected_rank) {
    return [=](ngraph::Output<ngraph::Node> output) -> bool {
        return ngraph::pattern::consumers_count(expected_count)(output) &&
               ngraph::pattern::rank_equals(expected_rank)(output);
    };
}

bool TransposeOrderMatches(std::shared_ptr<ngraph::opset7::Transpose> transpose, std::vector<size_t> order) {
    if (!transpose)
        return false;
    const ngraph::Output<ngraph::Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values =
        std::dynamic_pointer_cast<ngraph::opset7::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;

    const auto data = const_with_order_values->cast_vector<size_t>();
    if (data.empty())
        return false;

    if (!std::equal(order.begin(), order.end(), data.begin()))
        return false;

    return true;
}

std::shared_ptr<ngraph::opset7::StridedSlice> FlatCrop(ngraph::Output<ngraph::Node> input, size_t offset, size_t size) {
    return std::make_shared<ngraph::opset7::StridedSlice>(
        input,  // data
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         {(size_t)0, offset}),  // begin sice index
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         {(size_t)0, offset + size}),  // end slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)1, (size_t)1}),  // strides
        std::vector<int64_t>{1, 0},                                                                        // begin mask
        std::vector<int64_t>{1, 0});                                                                       // end mask
}

std::shared_ptr<ngraph::Node> VerifyBiasGetConst(std::shared_ptr<ngraph::Node> conv,
                                                 std::shared_ptr<ngraph::Node> add) {
    auto add_const = std::dynamic_pointer_cast<ngraph::opset7::Constant>(add->input_value(1).get_node_shared_ptr());

    // Check if it's really a bias and not just addition
    if (add_const) {
        auto bias_size = shape_size(add_const->get_shape());
        auto conv_filter_count = conv->get_output_shape(0)[1];
        if (bias_size == conv_filter_count)
            return add_const;
    }
    return nullptr;
}

std::shared_ptr<ngraph::Node> InsertFQLayer(const std::shared_ptr<ngraph::opset7::FakeQuantize> fq_layer,
                                            std::shared_ptr<ngraph::Node> last_node) {
    if (fq_layer != nullptr) {
        auto new_fq = fq_layer->clone_with_new_inputs(
            {last_node,
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(1).get_node_shared_ptr())
                     ->cast_vector<float>()),
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(2).get_node_shared_ptr())
                     ->cast_vector<float>()),
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(3).get_node_shared_ptr())
                     ->cast_vector<float>()),
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(4).get_node_shared_ptr())
                     ->cast_vector<float>())});
        ngraph::copy_runtime_info(new_fq, fq_layer);
        return new_fq;
    }
    return last_node;
}

void RemoveSingleInputNodeFromFunction(std::shared_ptr<ov::Node> node) {
    const ov::Shape input_node_shape = node->get_input_shape(0);
    const ov::Shape output_node_shape = node->get_output_shape(0);

    std::shared_ptr<ov::Node> node_parent = node->get_input_node_shared_ptr(0);
    if (!std::equal(input_node_shape.begin(), input_node_shape.end(), output_node_shape.begin())) {
        auto reshape_const_node =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{output_node_shape.size()}, output_node_shape);
        node_parent = std::make_shared<Reshape>(node_parent, reshape_const_node, false);
    }

    ov::replace_output_update_name(node->output(0), node_parent->output(0));
}

ov::Shape SqueezeShape(const ov::Shape& shape) {
    ov::Shape squeezed_shape;
    std::copy_if(shape.begin(), shape.end(), std::back_inserter(squeezed_shape), [](size_t x) {
        return x != 1;
    });
    return squeezed_shape;
}

}  // namespace helper
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
