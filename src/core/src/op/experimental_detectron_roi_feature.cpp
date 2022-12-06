// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_roi_feature.hpp"

#include <algorithm>
#include <experimental_detectron_roi_feature_shape_inference.hpp>
#include <memory>
#include <utility>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v6::ExperimentalDetectronROIFeatureExtractor);

op::v6::ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor(const OutputVector& args,
                                                                                           const Attributes& attrs)
    : Op(args),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

op::v6::ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor(const NodeVector& args,
                                                                                           const Attributes& attrs)
    : ExperimentalDetectronROIFeatureExtractor(as_output_vector(args), attrs) {}

bool op::v6::ExperimentalDetectronROIFeatureExtractor::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_visit_attributes);
    visitor.on_attribute("output_size", m_attrs.output_size);
    visitor.on_attribute("sampling_ratio", m_attrs.sampling_ratio);
    visitor.on_attribute("pyramid_scales", m_attrs.pyramid_scales);
    visitor.on_attribute("aligned", m_attrs.aligned);
    return true;
}

void op::v6::ExperimentalDetectronROIFeatureExtractor::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() >= 2, "At least two argument required.");

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}, ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes;
    for (size_t i = 0; i < get_input_size(); i++)
        input_shapes.push_back(get_input_partial_shape(i));

    shape_infer(this, input_shapes, output_shapes);

    auto input_et = get_input_element_type(0);

    set_output_size(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); i++)
        set_output_type(i, input_et, output_shapes[i]);
}

shared_ptr<Node> op::v6::ExperimentalDetectronROIFeatureExtractor::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(new_args, m_attrs);
}
