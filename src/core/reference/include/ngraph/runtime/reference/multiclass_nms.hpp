// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/multiclass_nms_base.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace multiclass_nms_impl {
struct InfoForNMS {
    Shape selected_outputs_shape;
    Shape selected_indices_shape;
    Shape selected_numrois_shape;
    Shape boxes_shape;
    Shape scores_shape;
    Shape roisnum_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    std::vector<int64_t> roisnum_data;
    size_t selected_outputs_shape_size;
    size_t selected_indices_shape_size;
    size_t selected_numrois_shape_size;
};

InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::util::MulticlassNmsBase>& nms,
                                 const std::vector<std::shared_ptr<HostTensor>>& inputs);
}  // namespace multiclass_nms_impl

void multiclass_nms(const float* boxes_data,
                    const Shape& boxes_data_shape,
                    const float* scores_data,
                    const Shape& scores_data_shape,
                    const int64_t* roisnum_data,
                    const Shape& roisnum_data_shape,
                    const op::util::MulticlassNmsBase::Attributes& attrs,
                    float* selected_outputs,
                    const Shape& selected_outputs_shape,
                    int64_t* selected_indices,
                    const Shape& selected_indices_shape,
                    int64_t* valid_outputs);

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
