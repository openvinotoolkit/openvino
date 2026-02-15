// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <sparse_fill_empty_rows_inst.h>
#include <json_object.h>

#include <sstream>

#include "sparse_fill_empty_rows_shape_inference.hpp"
#include "memory_accessor.hpp"
#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(sparse_fill_empty_rows)

SparseFillEmptyRows_inst::typed_primitive_inst(network& network, SparseFillEmptyRows_node const& node) : parent(network, node) {}

layout SparseFillEmptyRows_inst::calc_output_layout(SparseFillEmptyRows_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> SparseFillEmptyRows_inst::calc_output_layouts(SparseFillEmptyRows_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<sparse_fill_empty_rows>();

    const auto& values_layout = impl_param.get_input_layout(0);
    const auto& dense_shape_layout = impl_param.get_input_layout(1);
    const auto& indices_layout = impl_param.get_input_layout(2);
    const auto& default_value_layout = impl_param.get_input_layout(3);

    std::vector<ShapeType> input_shapes = {
        values_layout.get<ShapeType>(),
        dense_shape_layout.get<ShapeType>(),
        indices_layout.get<ShapeType>(),
        default_value_layout.get<ShapeType>(),
    };
    std::unordered_map<size_t, ov::Tensor> tensors;
    if (!primitive->dense_shape.empty()) {
        tensors.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{primitive->dense_shape.size()}, primitive->dense_shape.data()));
    }
    if (!primitive->indices.empty()) {
        tensors.emplace(2, ov::Tensor(ov::element::i64, ov::Shape{primitive->indices.size()}, primitive->indices.data()));
    }

    const auto ta = MemoryAccessor(&impl_param.memory_deps, impl_param.get_stream(), ov::make_tensor_accessor(tensors));

    std::vector<ShapeType> output_shapes;
    ov::op::v16::SparseFillEmptyRows op;
    output_shapes = shape_infer(&op, input_shapes, ta);

    return {
        layout{output_shapes[0], indices_layout.data_type, indices_layout.format},
        layout{output_shapes[1], values_layout.data_type, values_layout.format},
        layout{output_shapes[2], indices_layout.data_type, indices_layout.format}
    };
}

std::string SparseFillEmptyRows_inst::to_string(SparseFillEmptyRows_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite SparseFillEmptyRows_info;
    SparseFillEmptyRows_info.add("values", node.input(0).id());
    SparseFillEmptyRows_info.add("dense_shape", node.input(1).id());
    SparseFillEmptyRows_info.add("indices", node.input(2).id());
    SparseFillEmptyRows_info.add("default_value", node.input(3).id());
    node_info->add("SparseFillEmptyRows info", SparseFillEmptyRows_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
