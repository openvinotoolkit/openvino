// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <vector>

#include "one_hot_shape_inference.hpp"

namespace cldnn {
primitive_type_id one_hot::type_id() {
    static primitive_type_base<one_hot> instance;
    return &instance;
}

layout one_hot_inst::calc_output_layout(one_hot_node const& node) {
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();

    auto dt = desc->output_data_type ? *desc->output_data_type : input_layout.data_type;
    auto format = input_layout.format;

    if (desc->one_hot_axis > 4) {
        CLDNN_ERROR_MESSAGE(node.id(),
                            "Incorrect parameters configuration: one_hot_axis should be less or equal to 4.");
    }

    {
        ov::op::v1::OneHot op;
        try {
            // set_axis also calls resolve_axis method which tries to get input0 partial shape
            // thus wrap this call with try/catch.
            // it's safe as shape_infer method calls normalize_axis internally
            op.set_axis(desc->one_hot_axis);
        } catch (...) {}

        std::vector<ov::PartialShape> output_shapes = { ov::PartialShape::dynamic() };
        std::vector<ov::PartialShape> input_shapes = {
            input_layout.size,
            ov::PartialShape{},
            ov::PartialShape{},
            ov::PartialShape{}
        };

        int64_t depth = desc->depth;

        auto depth_tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::i64, ov::Shape{1}, static_cast<void*>(&depth));
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, depth_tensor}
        };
        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
        return {dt, layout::get_default_format(output_shapes[0].size()), output_shapes[0]};
    }
}

std::string one_hot_inst::to_string(one_hot_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    const auto& shape = desc->shape;
    const auto& one_hot_axis = desc->one_hot_axis;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite one_hot_info;
    one_hot_info.add("input id", input.id());
    one_hot_info.add("output shape", shape.to_string());
    one_hot_info.add("one-hot axis", one_hot_axis);

    node_info->add("one_hot info", one_hot_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

one_hot_inst::typed_primitive_inst(network& network, one_hot_node const& node) : parent(network, node) {
    // auto input_layout = node.input().get_output_layout();

    // const auto& output_sizes = argument.shape;
//
    // std::vector<tensor::value_type> input_dims = input_layout.get_dims();
    // std::vector<tensor::value_type> output_dims = {output_sizes.batch[0],
    //                                                output_sizes.feature[0],
    //                                                output_sizes.spatial[1],
    //                                                output_sizes.spatial[0]};

    // if (is_output_bfzyx(input_layout, node.get_primitive()->one_hot_axis)) {
    //     output_dims.insert(output_dims.begin() + 2, output_sizes.spatial[2]);
    // }

    // const auto& one_hot_axis = node.get_primitive()->one_hot_axis;

    // for (size_t i = 0, j = 0; j < output_dims.size() - 1; ++i, ++j) {
    //     if (j == one_hot_axis)
    //         ++j;
    //     if (input_dims[i] != output_dims[j]) {
    //         CLDNN_ERROR_MESSAGE(node.id(), "Incorrect parameters configuration: shape does not fit input size.");
    //     }
    // }
}
}  // namespace cldnn
