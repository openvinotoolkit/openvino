// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_prior_grid_generator.hpp"

#include "evaluate_node.hpp"

namespace experimental_prior_grid {
struct InfoForEDPriorGrid {
    ov::Shape output_shape;
    int64_t grid_h;
    int64_t grid_w;
    float stride_h;
    float stride_w;
};

constexpr size_t priors_port = 0;
constexpr size_t feature_map_port = 1;

ov::PartialShape infer_output_shape(const ov::TensorVector& inputs, bool flatten) {
    ov::PartialShape out_shape = {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4};

    if (flatten) {
        out_shape = ov::PartialShape{ov::Dimension::dynamic(), 4};
    }

    const auto priors_shape = inputs[priors_port].get_shape();
    const auto feature_map_shape = inputs[feature_map_port].get_shape();

    if (!ov::shape_size(priors_shape) || !ov::shape_size(feature_map_shape)) {
        return out_shape;
    }

    auto num_priors = priors_shape[0];
    auto featmap_height = feature_map_shape[2];
    auto featmap_width = feature_map_shape[3];

    if (flatten) {
        out_shape =
            ov::PartialShape{ov::Dimension(static_cast<int64_t>(featmap_height * featmap_width * num_priors)), 4};
    } else {
        out_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(featmap_height)),
                                     ov::Dimension(static_cast<int64_t>(featmap_width)),
                                     ov::Dimension(static_cast<int64_t>(num_priors)),
                                     4};
    }

    return out_shape;
}

InfoForEDPriorGrid get_info_for_ed_prior_grid_eval(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>& prior_grid,
    const ov::TensorVector& inputs) {
    InfoForEDPriorGrid result;

    auto attrs = prior_grid->get_attrs();

    result.grid_h = attrs.h;
    result.grid_w = attrs.w;
    result.stride_h = attrs.stride_y;
    result.stride_w = attrs.stride_x;

    auto output_rois_shape = infer_output_shape(inputs, attrs.flatten);
    result.output_shape = output_rois_shape.to_shape();

    return result;
}
}  // namespace experimental_prior_grid

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto info = experimental_prior_grid::get_info_for_ed_prior_grid_eval(op, inputs);

    using T = typename ov::element_type_traits<ET>::value_type;
    outputs[0].set_shape(info.output_shape);
    ov::reference::experimental_detectron_prior_grid_generator<T>(inputs[0].data<const T>(),
                                                                  inputs[0].get_shape(),
                                                                  inputs[1].get_shape(),
                                                                  inputs[2].get_shape(),
                                                                  outputs[0].data<T>(),
                                                                  info.grid_h,
                                                                  info.grid_w,
                                                                  info.stride_h,
                                                                  info.stride_w);

    return true;
}

template <>
bool evaluate_node<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(std::shared_ptr<ov::Node> node,
                                                                        ov::TensorVector& outputs,
                                                                        const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
            outputs,
            inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                           outputs,
                                           inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                         outputs,
                                         inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                         outputs,
                                         inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                         outputs,
                                         inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                         outputs,
                                         inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                         outputs,
                                         inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(node),
                                          outputs,
                                          inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
