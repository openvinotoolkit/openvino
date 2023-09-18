// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "evaluates_map.hpp"
#include "evaluate_node.hpp"
#include "openvino/reference/experimental_detectron_roi_feature_extractor.hpp"
// clang-format on

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

    ov::reference::experimental_detectron_roi_feature_extractor(input_data,
                                                                input_shapes,
                                                                attrs,
                                                                output_rois_features.data(),
                                                                output_rois.data());

    ov::reference::experimental_detectron_roi_feature_extractor_postprocessing(outputs[0]->get_data_ptr(),
                                                                               outputs[1]->get_data_ptr(),
                                                                               output_type,
                                                                               output_rois_features,
                                                                               output_rois,
                                                                               output_rois_features_shape,
                                                                               output_rois_shape);

    return true;
}

template <>
bool evaluate_node<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(std::shared_ptr<ngraph::Node> node,
                                                                             const ngraph::HostTensorVector& outputs,
                                                                             const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
