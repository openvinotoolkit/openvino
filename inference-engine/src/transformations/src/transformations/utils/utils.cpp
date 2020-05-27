// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/utils.hpp"

#include <assert.h>

#include <functional>
#include <memory>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/util/op_annotations.hpp>

namespace ngraph {
namespace op {
namespace util {

bool get_single_value(const std::shared_ptr<op::Constant>& const_node, float& value) {
    switch (const_node->get_element_type()) {
    case element::Type_t::f16:
        return util::normalize_single_value(const_node->get_vector<float16>(), value);
    case element::Type_t::f32:
        return util::normalize_single_value(const_node->get_vector<float>(), value);
    case element::Type_t::bf16:
        return util::normalize_single_value(const_node->get_vector<bfloat16>(), value);
    case element::Type_t::f64:
        return util::normalize_single_value(const_node->get_vector<double>(), value);
    case element::Type_t::i8:
        return util::normalize_single_value(const_node->get_vector<int8_t>(), value);
    case element::Type_t::i16:
        return util::normalize_single_value(const_node->get_vector<int16_t>(), value);
    case element::Type_t::i32:
        return util::normalize_single_value(const_node->get_vector<int32_t>(), value);
    case element::Type_t::i64:
        return util::normalize_single_value(const_node->get_vector<int64_t>(), value);
    case element::Type_t::u8:
        return util::normalize_single_value(const_node->get_vector<uint8_t>(), value);
    case element::Type_t::u16:
        return util::normalize_single_value(const_node->get_vector<uint16_t>(), value);
    case element::Type_t::u32:
        return util::normalize_single_value(const_node->get_vector<uint32_t>(), value);
    case element::Type_t::u64:
        return util::normalize_single_value(const_node->get_vector<uint64_t>(), value);
    default:
        throw ngraph_error("Unsupported precision for const operation: " + const_node->get_friendly_name());
    }
}

std::shared_ptr<Node> normalize_constant(const std::shared_ptr<op::Constant>& constant,
                                         const PartialShape& shape) {
    auto const_shape = constant->get_shape();
    if (const_shape.size() == shape.rank().get_length()) {
        return constant;
    }
    int64_t cnt = shape.rank().get_length() - const_shape.size();
    for (int i = 0; i < cnt; ++i) {
        const_shape.insert(const_shape.begin(), 1);
    }

    return reshapeTo(constant, const_shape);
}

std::shared_ptr<Node> broadcastTo(const Output<Node>& input, const ngraph::Shape& shape) {
    if (input.get_shape().size() != shape.size())
        throw ngraph_error("Shape dims mismatch");
    for (size_t i = 0; i < input.get_shape().size(); i++) {
        if (input.get_shape()[i] != 1 && input.get_shape()[i] != shape[i])
            throw ngraph_error("Shape mismatch");
    }
    return std::make_shared<op::v1::Broadcast>(input, op::Constant::create(ngraph::element::i64, Shape {shape.size()}, shape));
}

std::shared_ptr<ngraph::Node> reshapeTo(const Output<Node> & input, const Shape& shape) {
    return std::make_shared<op::v1::Reshape>(input, op::Constant::create(element::i64, Shape{shape.size()}, shape), true);
}

bool constantIsEqualTo(const std::shared_ptr<ngraph::op::Constant>& const_node, float value, float eps) {
    float res(0);
    if (!get_single_value(const_node, res)) {
        return false;
    }

    return std::abs(res - value) < eps;
}

bool has_f16_constants(const std::shared_ptr<const ngraph::Function> &function) {
    for (auto & layer : function->get_ops()) {
        if (std::dynamic_pointer_cast<ngraph::op::Constant>(layer) && layer->output(0).get_element_type() == ngraph::element::f16) {
            return true;
        }
    }
    return false;
}

}  // namespace util
}  // namespace op
}  // namespace ngraph
