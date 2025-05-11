// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits.h>

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
static std::shared_ptr<Node> get_val(int32_t idx, const Output<Node>& data) {
    auto startsNode = ov::opset6::Constant::create(element::i32, {1}, {idx});
    auto endsNode = ov::opset6::Constant::create(element::i32, {1}, {idx + 1});
    auto stridesNode = ov::opset6::Constant::create(element::i32, {1}, {1});
    return std::make_shared<ov::opset6::StridedSlice>(data,
                                                      startsNode,
                                                      endsNode,
                                                      stridesNode,
                                                      std::vector<int64_t>(1, 0),
                                                      std::vector<int64_t>(1, 0));
}

static std::shared_ptr<Node> set_val(int32_t idx, std::shared_ptr<Node> val_node, std::shared_ptr<Node> array_node) {
    NodeVector nodes;
    if (idx > 0) {
        // [0, idx)
        auto startsNode = ov::opset6::Constant::create(element::i32, {1}, {0});
        auto endsNode = ov::opset6::Constant::create(element::i32, {1}, {idx});
        auto stridesNode = ov::opset6::Constant::create(element::i32, {1}, {1});
        auto head = std::make_shared<ov::opset6::StridedSlice>(array_node,
                                                               startsNode,
                                                               endsNode,
                                                               stridesNode,
                                                               std::vector<int64_t>(1, 0),
                                                               std::vector<int64_t>(1, 0));
        nodes.push_back(head);
    }
    nodes.push_back(val_node);
    // [idx + 1, max)
    auto startsNode = ov::opset6::Constant::create(element::i32, {1}, {idx + 1});
    auto endsNode = ov::opset6::Constant::create(element::i32, {1}, {INT_MAX});
    auto stridesNode = ov::opset6::Constant::create(element::i32, {1}, {1});
    auto tail = std::make_shared<ov::opset6::StridedSlice>(array_node,
                                                           startsNode,
                                                           endsNode,
                                                           stridesNode,
                                                           std::vector<int64_t>(1, 0),
                                                           std::vector<int64_t>(1, 0));
    nodes.push_back(tail);

    return std::make_shared<ov::opset6::Concat>(nodes, 0);
}

template <element::Type_t Type,
          typename StorageDataType = fundamental_type_for<Type>,
          typename std::enable_if<Type == element::Type_t::i32 || Type == element::Type_t::i64 ||
                                      Type == element::Type_t::f32 || Type == element::Type_t::f64,
                                  bool>::type = true>
static Output<Node> get_seed_node(const NodeContext& node) {
    Output<Node> val_node;
    auto dtype = node.get_attribute<element::Type>("dtype");
    auto str_value = node.get_attribute<std::string>("str_value");
    if (str_value.empty()) {
        auto float_value = node.get_attribute<float>("value");
        val_node = ov::opset6::Constant::create(dtype, {1}, {static_cast<StorageDataType>(float_value)});
    } else {
        std::stringstream ss(str_value);
        StorageDataType tmp_value;
        ss >> tmp_value;
        val_node = ov::opset6::Constant::create(dtype, {1}, {static_cast<StorageDataType>(tmp_value)});
    }
    return val_node;
}

static Output<Node> get_seed_node(const NodeContext& node) {
    Output<Node> val_node;
    auto dtype = node.get_attribute<element::Type>("dtype");

    switch (dtype) {
    case element::i32:
        val_node = get_seed_node<element::i32>(node);
        break;
    case element::i64:
        val_node = get_seed_node<element::i64>(node);
        break;
    case element::f32:
        val_node = get_seed_node<element::f32>(node);
        break;
    case element::f64:
        val_node = get_seed_node<element::f64>(node);
        break;
    default:
        throw std::runtime_error("fill_constant_batch_size_like: unsupported dtype");
    }

    return val_node;
}

NamedOutputs fill_constant_batch_size_like(const NodeContext& node) {
    auto input_dim_idx = node.get_attribute<int32_t>("input_dim_idx");
    auto output_dim_idx = node.get_attribute<int32_t>("output_dim_idx");
    auto shapes = node.get_attribute<std::vector<int32_t>>("shape");
    auto input = node.get_input("Input");
    auto input_shape = std::make_shared<ov::opset6::ShapeOf>(input, element::i32);
    // 1, cat the array:
    //   shape[0, shape[output_dim_idx]) + input_shape[input_dim_idx] +
    //   shape[shape[output_dim_idx + 1], -1]
    auto input_val_node = get_val(input_dim_idx, input_shape);
    auto shapes_node = ov::opset6::Constant::create(ov::element::i32, {shapes.size()}, shapes);
    auto shape_node = set_val(output_dim_idx, input_val_node, shapes_node);

    // 2, use the shape broadcast the node
    auto val_node = get_seed_node(node);
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Broadcast>(val_node, shape_node)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov