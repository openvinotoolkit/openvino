// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/utils/nms_common.hpp"
#include "ngraph/runtime/reference/multiclass_nms.hpp"
#include "openvino/op/multiclass_nms.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::MulticlassNms>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = ngraph::runtime::reference::multiclass_nms_impl::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.selected_numrois_shape_size);

    ngraph::runtime::reference::multiclass_nms(info.boxes_data.data(),
                                       info.boxes_shape,
                                       info.scores_data.data(),
                                       info.scores_shape,
                                       nullptr,
                                       ov::Shape(),  // won't be used
                                       op->get_attrs(),
                                       selected_outputs.data(),
                                       info.selected_outputs_shape,
                                       selected_indices.data(),
                                       info.selected_indices_shape,
                                       valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), int64_t(0)));

    outputs[0]->set_shape({num_selected, 6});
    prois = outputs[0]->get_data_ptr();

    if (outputs.size() >= 2) {
        outputs[1]->set_shape({num_selected, 1});
        pscores = outputs[1]->get_data_ptr();
    }
    if (outputs.size() >= 3) {
        pselected_num = outputs[2]->get_data_ptr();
    }

    ngraph::runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                              pscores,
                                                              pselected_num,
                                                              op->get_attrs().output_type,
                                                              selected_outputs,
                                                              selected_indices,
                                                              valid_outputs,
                                                              op->get_input_element_type(0));

    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v9::MulticlassNms>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = ngraph::runtime::reference::multiclass_nms_impl::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.selected_numrois_shape_size);

    ngraph::runtime::reference::multiclass_nms(info.boxes_data.data(),
                                       info.boxes_shape,
                                       info.scores_data.data(),
                                       info.scores_shape,
                                       info.roisnum_data.data(),
                                       info.roisnum_shape,
                                       op->get_attrs(),
                                       selected_outputs.data(),
                                       info.selected_outputs_shape,
                                       selected_indices.data(),
                                       info.selected_indices_shape,
                                       valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0));

    outputs[0]->set_shape({num_selected, 6});
    prois = outputs[0]->get_data_ptr();

    if (outputs.size() >= 2) {
        outputs[1]->set_shape({num_selected, 1});
        pscores = outputs[1]->get_data_ptr();
    }
    if (outputs.size() >= 3) {
        pselected_num = outputs[2]->get_data_ptr();
    }

    ngraph::runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                              pscores,
                                                              pselected_num,
                                                              op->get_attrs().output_type,
                                                              selected_outputs,
                                                              selected_indices,
                                                              valid_outputs,
                                                              op->get_input_element_type(0));

    return true;
}
