// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_roi_feature.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "experimental_detectron_roi_feature_shape_inference.hpp"
#include "experimental_detectron_shape_infer_utils.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

namespace ov {
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

    const auto shapes_and_type = detectron::validate::all_inputs_same_floating_type(this);
    const auto output_shapes = shape_infer(this, shapes_and_type.first);

    for (size_t i = 0; i < output_shapes.size(); i++)
        set_output_type(i, shapes_and_type.second, output_shapes[i]);
}

std::shared_ptr<Node> op::v6::ExperimentalDetectronROIFeatureExtractor::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(new_args, m_attrs);
}

void op::v6::ExperimentalDetectronROIFeatureExtractor::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace ov
