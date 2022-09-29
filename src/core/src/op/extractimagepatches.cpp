// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/extractimagepatches.hpp"

#include <extract_image_patches_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

// ExtractImagePatches v3

BWDCMP_RTTI_DEFINITION(op::v3::ExtractImagePatches);

op::v3::ExtractImagePatches::ExtractImagePatches(const Output<Node>& image,
                                                 const ov::Shape& sizes,
                                                 const Strides& strides,
                                                 const ov::Shape& rates,
                                                 const PadType& auto_pad)
    : Op({image}),
      m_patch_sizes(sizes),
      m_patch_movement_strides(strides),
      m_patch_selection_rates(rates),
      m_padding(auto_pad) {
    constructor_validate_and_infer_types();
}

void op::v3::ExtractImagePatches::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ExtractImagePatches_validate_and_infer_types);
    std::vector<PartialShape> input_shapes = {get_input_partial_shape(0)}, output_shapes = {PartialShape{}};
    shape_infer(this, input_shapes, output_shapes);
    if (output_shapes[0].is_dynamic())
        set_input_is_relevant_to_shape(0);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool op::v3::ExtractImagePatches::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ExtractImagePatches_visit_attributes);
    visitor.on_attribute("sizes", m_patch_sizes);
    visitor.on_attribute("strides", m_patch_movement_strides);
    visitor.on_attribute("rates", m_patch_selection_rates);
    visitor.on_attribute("auto_pad", m_padding);
    return true;
}

shared_ptr<Node> op::v3::ExtractImagePatches::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ExtractImagePatches_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v3::ExtractImagePatches>(new_args.at(0),
                                                    m_patch_sizes,
                                                    m_patch_movement_strides,
                                                    m_patch_selection_rates,
                                                    m_padding);
}
