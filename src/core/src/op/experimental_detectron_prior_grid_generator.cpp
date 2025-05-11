// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"

#include <memory>

#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "experimental_detectron_shape_infer_utils.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

namespace ov {
op::v6::ExperimentalDetectronPriorGridGenerator::ExperimentalDetectronPriorGridGenerator(
    const Output<Node>& priors,
    const Output<Node>& feature_map,
    const Output<Node>& im_data,
    const Attributes& attrs)
    : Op({priors, feature_map, im_data}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronPriorGridGenerator::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronPriorGridGenerator_visit_attributes);
    visitor.on_attribute("flatten", m_attrs.flatten);
    visitor.on_attribute("h", m_attrs.h);
    visitor.on_attribute("w", m_attrs.w);
    visitor.on_attribute("stride_x", m_attrs.stride_x);
    visitor.on_attribute("stride_y", m_attrs.stride_y);
    return true;
}

std::shared_ptr<Node> op::v6::ExperimentalDetectronPriorGridGenerator::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronPriorGridGenerator_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ExperimentalDetectronPriorGridGenerator>(new_args.at(0),
                                                                     new_args.at(1),
                                                                     new_args.at(2),
                                                                     m_attrs);
}

void op::v6::ExperimentalDetectronPriorGridGenerator::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronPriorGridGenerator_validate_and_infer_types);

    const auto shapes_and_type = detectron::validate::all_inputs_same_floating_type(this);
    const auto output_shapes = shape_infer(this, shapes_and_type.first);

    set_output_type(0, shapes_and_type.second, output_shapes[0]);
}

void op::v6::ExperimentalDetectronPriorGridGenerator::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace ov
