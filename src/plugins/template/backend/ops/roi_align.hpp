// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/roi_align.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v9::ROIAlign>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    std::vector<int64_t> batch_indices_vec_scaled_up = host_tensor_2_vector<int64_t>(inputs[2]);
    ngraph::op::v3::ROIAlign::PoolingMode m_mode_v3;
    switch (op->get_mode()) {
    case ngraph::op::v9::ROIAlign::PoolingMode::AVG: {
        m_mode_v3 = ngraph::op::v3::ROIAlign::PoolingMode::AVG;
        break;
    }
    case ngraph::op::v9::ROIAlign::PoolingMode::MAX: {
        m_mode_v3 = ngraph::op::v3::ROIAlign::PoolingMode::MAX;
        break;
    }
    default: {
        NGRAPH_CHECK(false, "unsupported PoolingMode ");
    }
    }
    ngraph::runtime::reference::roi_align<T>(inputs[0]->get_data_ptr<const T>(),
                                             inputs[1]->get_data_ptr<const T>(),
                                             batch_indices_vec_scaled_up.data(),
                                             outputs[0]->get_data_ptr<T>(),
                                             op->get_input_shape(0),
                                             op->get_input_shape(1),
                                             op->get_input_shape(2),
                                             op->get_output_shape(0),
                                             op->get_pooled_h(),
                                             op->get_pooled_w(),
                                             op->get_sampling_ratio(),
                                             op->get_spatial_scale(),
                                             m_mode_v3,
                                             op->get_aligned_mode());
    return true;
}