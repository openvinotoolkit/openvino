// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_roi_feature_extractor.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

namespace experimental_roi_feature {
struct InfoForEDROIFeature {
    ov::Shape output_rois_features_shape;
    ov::Shape output_rois_shape;
};

InfoForEDROIFeature get_info_for_ed_roi_feature(
    const std::vector<ov::Shape> input_shapes,
    const ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs) {
    InfoForEDROIFeature result;

    size_t output_size = static_cast<size_t>(attrs.output_size);
    auto out_shape = ov::Shape{0, 0, output_size, output_size};
    auto out_rois_shape = ov::Shape{0, 4};

    auto rois_shape = input_shapes[0];

    out_shape[0] = rois_shape[0];
    out_rois_shape[0] = rois_shape[0];

    out_shape[1] = input_shapes[1][1];

    result.output_rois_features_shape = out_shape;
    result.output_rois_shape = out_rois_shape;

    return result;
}
}  // namespace experimental_roi_feature

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    const auto attrs = op->get_attrs();

    std::vector<std::vector<float>> input_data;
    std::vector<ov::Shape> input_shapes;
    for (const auto& input : inputs) {
        const auto current_shape = input.get_shape();
        input_data.push_back(get_floats(input, current_shape));
        input_shapes.push_back(current_shape);
    }

    const auto info = experimental_roi_feature::get_info_for_ed_roi_feature(input_shapes, attrs);
    const auto& output_rois_features_shape = info.output_rois_features_shape;
    const auto& output_rois_shape = info.output_rois_shape;

    const auto output_type = op->get_input_element_type(0);

    outputs[0].set_shape(output_rois_features_shape);
    outputs[1].set_shape(output_rois_shape);

    std::vector<float> output_rois_features(ov::shape_size(output_rois_features_shape));
    std::vector<float> output_rois(ov::shape_size(output_rois_shape));

    ov::reference::experimental_detectron_roi_feature_extractor(input_data,
                                                                input_shapes,
                                                                attrs,
                                                                output_rois_features.data(),
                                                                output_rois.data());

    ov::reference::experimental_detectron_roi_feature_extractor_postprocessing(outputs[0].data(),
                                                                               outputs[1].data(),
                                                                               output_type,
                                                                               output_rois_features,
                                                                               output_rois,
                                                                               output_rois_features_shape,
                                                                               output_rois_shape);

    return true;
}

template <>
bool evaluate_node<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(std::shared_ptr<ov::Node> node,
                                                                         ov::TensorVector& outputs,
                                                                         const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
            outputs,
            inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                           outputs,
                                           inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                         outputs,
                                         inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                         outputs,
                                         inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                         outputs,
                                         inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                         outputs,
                                         inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                         outputs,
                                         inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(node),
                                          outputs,
                                          inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
