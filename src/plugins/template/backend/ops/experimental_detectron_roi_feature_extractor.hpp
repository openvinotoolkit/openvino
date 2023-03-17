// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/experimental_detectron_roi_feature_extractor.hpp"

namespace experimental_roi_feature {
struct InfoForEDROIFeature {
    ngraph::Shape output_rois_features_shape;
    ngraph::Shape output_rois_shape;
};

InfoForEDROIFeature get_info_for_ed_roi_feature(
    const std::vector<ngraph::Shape> input_shapes,
    const ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs) {
    InfoForEDROIFeature result;

    size_t output_size = static_cast<size_t>(attrs.output_size);
    auto out_shape = ngraph::Shape{0, 0, output_size, output_size};
    auto out_rois_shape = ngraph::Shape{0, 4};

    auto rois_shape = input_shapes[0];

    out_shape[0] = rois_shape[0];
    out_rois_shape[0] = rois_shape[0];

    out_shape[1] = input_shapes[1][1];

    result.output_rois_features_shape = out_shape;
    result.output_rois_shape = out_rois_shape;

    return result;
}
}  // namespace experimental_roi_feature

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto attrs = op->get_attrs();

    std::vector<std::vector<float>> input_data;
    std::vector<ngraph::Shape> input_shapes;
    for (const auto& input : inputs) {
        const auto current_shape = input->get_shape();
        input_data.push_back(get_floats(input, current_shape));
        input_shapes.push_back(current_shape);
    }

    const auto info = experimental_roi_feature::get_info_for_ed_roi_feature(input_shapes, attrs);
    const auto& output_rois_features_shape = info.output_rois_features_shape;
    const auto& output_rois_shape = info.output_rois_shape;

    const auto output_type = op->get_input_element_type(0);

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(output_rois_features_shape);
    outputs[1]->set_element_type(output_type);
    outputs[1]->set_shape(output_rois_shape);

    std::vector<float> output_rois_features(ngraph::shape_size(output_rois_features_shape));
    std::vector<float> output_rois(ngraph::shape_size(output_rois_shape));

    ngraph::runtime::reference::experimental_detectron_roi_feature_extractor(input_data,
                                                                             input_shapes,
                                                                             attrs,
                                                                             output_rois_features.data(),
                                                                             output_rois.data());

    ngraph::runtime::reference::experimental_detectron_roi_feature_extractor_postprocessing(outputs[0]->get_data_ptr(),
                                                                                            outputs[1]->get_data_ptr(),
                                                                                            output_type,
                                                                                            output_rois_features,
                                                                                            output_rois,
                                                                                            output_rois_features_shape,
                                                                                            output_rois_shape);

    return true;
}