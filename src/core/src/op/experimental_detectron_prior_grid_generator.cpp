// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_prior_grid_generator.hpp"

#include <experimental_detectron_prior_grid_generator_shape_inference.hpp>
#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v6::ExperimentalDetectronPriorGridGenerator);

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

shared_ptr<Node> op::v6::ExperimentalDetectronPriorGridGenerator::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronPriorGridGenerator_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronPriorGridGenerator>(new_args.at(0),
                                                                        new_args.at(1),
                                                                        new_args.at(2),
                                                                        m_attrs);
}

static constexpr size_t priors_port = 0;
static constexpr size_t featmap_port = 1;
static constexpr size_t im_data_port = 2;

void op::v6::ExperimentalDetectronPriorGridGenerator::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronPriorGridGenerator_validate_and_infer_types);
    const auto& priors_shape = get_input_partial_shape(priors_port);
    const auto& featmap_shape = get_input_partial_shape(featmap_port);
    const auto& input_et = get_input_element_type(0);

    set_output_size(1);
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {priors_shape, featmap_shape, get_input_partial_shape(im_data_port)};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, input_et, output_shapes[0]);
}
