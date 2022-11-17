// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/region_yolo.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "region_yolo_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::RegionYolo);

op::RegionYolo::RegionYolo(const Output<Node>& input,
                           const size_t coords,
                           const size_t classes,
                           const size_t regions,
                           const bool do_softmax,
                           const vector<int64_t>& mask,
                           const int axis,
                           const int end_axis,
                           const vector<float>& anchors)
    : Op({input}),
      m_num_coords(coords),
      m_num_classes(classes),
      m_num_regions(regions),
      m_do_softmax(do_softmax),
      m_mask(mask),
      m_anchors(anchors),
      m_axis(axis),
      m_end_axis(end_axis) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::RegionYolo::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_RegionYolo_visit_attributes);
    visitor.on_attribute("anchors", m_anchors);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("coords", m_num_coords);
    visitor.on_attribute("classes", m_num_classes);
    visitor.on_attribute("end_axis", m_end_axis);
    visitor.on_attribute("num", m_num_regions);
    visitor.on_attribute("do_softmax", m_do_softmax);
    visitor.on_attribute("mask", m_mask);
    return true;
}

void op::RegionYolo::validate_and_infer_types() {
    OV_OP_SCOPE(v0_RegionYolo_validate_and_infer_types);
    auto input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_real(),
                          "Type of input is expected to be a floating point type. Got: ",
                          input_et);
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, input_et, output_shapes[0]);
}

shared_ptr<Node> op::RegionYolo::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_RegionYolo_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<RegionYolo>(new_args.at(0),
                                   m_num_coords,
                                   m_num_classes,
                                   m_num_regions,
                                   m_do_softmax,
                                   m_mask,
                                   m_axis,
                                   m_end_axis,
                                   m_anchors);
}
