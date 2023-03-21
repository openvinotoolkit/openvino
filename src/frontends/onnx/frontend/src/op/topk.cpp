// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/topk.hpp"

#include <cstdint>
#include <memory>
#include <openvino/opsets/opset11.hpp>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/validation_util.hpp"
#include "op/topk.hpp"
#include "utils/reshape.hpp"

namespace {
/// \return Return the second input to the TopK node reshaped to a scalar.
ngraph::Output<ngraph::Node> get_k(const ngraph::onnx_import::Node& node) {
    auto k_node = node.get_ng_inputs().at(1);
    NGRAPH_CHECK(shape_size(k_node.get_shape()) == 1,
                 "ONNX TopK operator: 'K' parameter must contain a single positive value.",
                 node);

    return ngraph::onnx_import::reshape::interpret_as_scalar(k_node);
}
}  // namespace

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector topk(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    const auto k_node = node.get_attribute_as_constant<std::int64_t>("k");
    const std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};

    std::shared_ptr<ngraph::Node> top_k = std::make_shared<ov::opset11::TopK>(data,
                                                                              k_node,
                                                                              axis,
                                                                              ov::opset11::TopK::Mode::MAX,
                                                                              ov::opset11::TopK::SortType::SORT_VALUES,
                                                                              element::i64);

    return {top_k->output(0), top_k->output(1)};
}
}  // namespace set_1

namespace set_10 {
OutputVector topk(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto k = get_k(node);
    const std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};

    std::shared_ptr<ngraph::Node> top_k = std::make_shared<ov::opset11::TopK>(data,
                                                                              k,
                                                                              axis,
                                                                              ov::opset11::TopK::Mode::MAX,
                                                                              ov::opset11::TopK::SortType::SORT_VALUES,
                                                                              element::i64);

    return {top_k->output(0), top_k->output(1)};
}
}  // namespace set_10

namespace set_11 {
OutputVector topk(const Node& node) {
    // Process inputs
    auto data = node.get_ng_inputs().at(0);
    auto k = get_k(node);

    // Process attributes
    const std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};
    const auto largest = node.get_attribute_value<std::int64_t>("largest", 1);
    const auto sorted = node.get_attribute_value<std::int64_t>("sorted", 1);

    // Map attribute values to nGraph enums
    const auto sort_type = sorted ? ov::opset11::TopK::SortType::SORT_VALUES : ov::opset11::TopK::SortType::NONE;

    const auto compute_max = static_cast<bool>(largest);
    const auto mode = compute_max ? ov::opset11::TopK::Mode::MAX : ov::opset11::TopK::Mode::MIN;

    std::shared_ptr<ngraph::Node> top_k =
        std::make_shared<ov::opset11::TopK>(data, k, axis, mode, sort_type, element::i64);

    return {top_k->output(0), top_k->output(1)};
}
}  // namespace set_11

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
