// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/experimental_detectron_prior_grid_generator.hpp"
#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"

struct InfoForEDPriorGrid {
    ov::Shape output_shape;
    int64_t grid_h;
    int64_t grid_w;
    float stride_h;
    float stride_w;
};

constexpr size_t priors_port = 0;
constexpr size_t feature_map_port = 1;

ov::PartialShape infer_output_shape(const std::vector<std::shared_ptr<ov::HostTensor>>& inputs, bool flatten) {
    ov::PartialShape out_shape = {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4};

    if (flatten) {
        out_shape = ov::PartialShape{ov::Dimension::dynamic(), 4};
    }

    const auto priors_shape = inputs[priors_port]->get_partial_shape();
    const auto feature_map_shape = inputs[feature_map_port]->get_partial_shape();

    if (priors_shape.rank().is_dynamic() || feature_map_shape.rank().is_dynamic()) {
        return out_shape;
    }

    auto num_priors = priors_shape[0];
    auto featmap_height = feature_map_shape[2];
    auto featmap_width = feature_map_shape[3];

    if (flatten) {
        out_shape = ov::PartialShape{featmap_height * featmap_width * num_priors, 4};
    } else {
        out_shape = ov::PartialShape{featmap_height, featmap_width, num_priors, 4};
    }

    return out_shape;
}

InfoForEDPriorGrid get_info_for_ed_prior_grid_eval(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>& prior_grid,
    const std::vector<std::shared_ptr<ov::HostTensor>>& inputs) {
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

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = get_info_for_ed_prior_grid_eval(op, inputs);

    using T = typename ov::element_type_traits<ET>::value_type;
    outputs[0]->set_shape(info.output_shape);
    ngraph::runtime::reference::experimental_detectron_prior_grid_generator<T>(inputs[0]->get_data_ptr<const T>(),
                                                                       inputs[0]->get_shape(),
                                                                       inputs[1]->get_shape(),
                                                                       inputs[2]->get_shape(),
                                                                       outputs[0]->get_data_ptr<T>(),
                                                                       info.grid_h,
                                                                       info.grid_w,
                                                                       info.stride_h,
                                                                       info.stride_w);

    return true;
}
