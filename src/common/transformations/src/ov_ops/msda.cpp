// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/msda.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

MSDA::MSDA(const Output<Node>& value,
           const Output<Node>& value_spatial_shapes,
           const Output<Node>& level_start_index,
           const Output<Node>& sampling_locations,
           const Output<Node>& attention_weights)
    : Op({value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MSDA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MSDA_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<MSDA>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}

bool MSDA::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MSDA_visit_attributes);
    return true;
}

// The inputs and the output are described in ov_ops/msda.hpp.
void MSDA::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MSDA_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 5, "MSDA must have 5 inputs whereas it has ", get_input_size());

    // value, sampling_locations and attention_weights share one floating type;
    // value_spatial_shapes and level_start_index are integer.
    auto data_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          ov::element::Type::merge(data_type, data_type, get_input_element_type(3)) &&
                              ov::element::Type::merge(data_type, data_type, get_input_element_type(4)),
                          "MSDA value, sampling_locations and attention_weights must have the same element type, got ",
                          get_input_element_type(0),
                          ", ",
                          get_input_element_type(3),
                          " and ",
                          get_input_element_type(4));
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type.is_real(),
                          "MSDA value must have a floating point element type, got ",
                          data_type);
    for (size_t i : {1, 2}) {
        const auto& index_type = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              index_type.is_dynamic() || index_type.is_integral_number(),
                              "MSDA value_spatial_shapes and level_start_index must have an integer element type, got ",
                              index_type);
    }

    const auto& value = get_input_partial_shape(0);
    const auto& spatial_shapes = get_input_partial_shape(1);
    const auto& level_start_index = get_input_partial_shape(2);
    const auto& locations = get_input_partial_shape(3);
    const auto& weights = get_input_partial_shape(4);
    NODE_VALIDATION_CHECK(this,
                          value.rank().compatible(4),
                          "MSDA value input must be 4D (bs, num_keys, num_heads, embed_dims), got ",
                          value);
    NODE_VALIDATION_CHECK(
        this,
        spatial_shapes.rank().compatible(2) && (spatial_shapes.rank().is_dynamic() || spatial_shapes[1].compatible(2)),
        "MSDA value_spatial_shapes input must have shape (num_levels, 2), got ",
        spatial_shapes);
    NODE_VALIDATION_CHECK(this,
                          level_start_index.rank().compatible(1),
                          "MSDA level_start_index input must be 1D (num_levels), got ",
                          level_start_index);
    NODE_VALIDATION_CHECK(
        this,
        locations.rank().compatible(6) && (locations.rank().is_dynamic() || locations[5].compatible(2)),
        "MSDA sampling_locations input must have shape (bs, num_queries, num_heads, num_levels, "
        "num_points, 2), got ",
        locations);
    NODE_VALIDATION_CHECK(
        this,
        weights.rank().compatible(5),
        "MSDA attention_weights input must be 5D (bs, num_queries, num_heads, num_levels, num_points), "
        "got ",
        weights);

    // Every dimension must agree between the inputs that carry it.
    ov::Dimension batch, num_queries, num_heads, embed_dims, num_levels, num_points;
    const auto merge = [this](ov::Dimension& dim, const ov::PartialShape& shape, size_t axis, const char* name) {
        NODE_VALIDATION_CHECK(this,
                              shape.rank().is_dynamic() || ov::Dimension::merge(dim, dim, shape[axis]),
                              "MSDA inputs have inconsistent ",
                              name,
                              ": value ",
                              get_input_partial_shape(0),
                              ", value_spatial_shapes ",
                              get_input_partial_shape(1),
                              ", level_start_index ",
                              get_input_partial_shape(2),
                              ", sampling_locations ",
                              get_input_partial_shape(3),
                              ", attention_weights ",
                              get_input_partial_shape(4));
    };
    for (const auto* shape : {&value, &locations, &weights})
        merge(batch, *shape, 0, "batch size");
    merge(num_queries, locations, 1, "number of queries");
    merge(num_queries, weights, 1, "number of queries");
    merge(num_heads, value, 2, "number of heads");
    merge(num_heads, locations, 2, "number of heads");
    merge(num_heads, weights, 2, "number of heads");
    merge(embed_dims, value, 3, "number of channels");
    merge(num_levels, spatial_shapes, 0, "number of levels");
    merge(num_levels, level_start_index, 0, "number of levels");
    merge(num_levels, locations, 3, "number of levels");
    merge(num_levels, weights, 3, "number of levels");
    merge(num_points, locations, 4, "number of points");
    merge(num_points, weights, 4, "number of points");

    set_output_type(0, data_type, ov::PartialShape{batch, num_queries, num_heads * embed_dims});
}

}  // namespace internal
}  // namespace op
}  // namespace ov
