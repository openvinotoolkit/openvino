// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/deformable_psroi_pooling.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/deformable_psroi_pooling.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::DeformablePSROIPooling>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    NGRAPH_CHECK(inputs.size() > 1 && inputs[1]->get_shape().size() == 2,
                 "2D tensor must be provided as second input. ");
    outputs[0]->set_shape({inputs[1]->get_shape()[0],
                           static_cast<size_t>(op->get_output_dim()),
                           static_cast<size_t>(op->get_group_size()),
                           static_cast<size_t>(op->get_group_size())});

    const bool has_offset_intput = inputs.size() == 3;
    if (has_offset_intput) {
        ngraph::runtime::reference::deformable_psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                                inputs[0]->get_shape(),
                                                                inputs[1]->get_data_ptr<T>(),
                                                                inputs[1]->get_shape(),
                                                                inputs[2]->get_data_ptr<T>(),
                                                                inputs[2]->get_shape(),
                                                                outputs[0]->get_data_ptr<T>(),
                                                                outputs[0]->get_shape(),
                                                                op->get_mode(),
                                                                op->get_spatial_scale(),
                                                                op->get_spatial_bins_x(),
                                                                op->get_spatial_bins_y(),
                                                                op->get_trans_std(),
                                                                op->get_part_size());
    } else {
        ngraph::runtime::reference::deformable_psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                                inputs[0]->get_shape(),
                                                                inputs[1]->get_data_ptr<T>(),
                                                                inputs[1]->get_shape(),
                                                                nullptr,
                                                                ngraph::Shape(),
                                                                outputs[0]->get_data_ptr<T>(),
                                                                outputs[0]->get_shape(),
                                                                op->get_mode(),
                                                                op->get_spatial_scale(),
                                                                op->get_spatial_bins_x(),
                                                                op->get_spatial_bins_y(),
                                                                op->get_trans_std(),
                                                                op->get_part_size());
    }
    return true;
}
