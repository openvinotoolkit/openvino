// Copyright (C) 2018-2021 Intel Corporation
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
#include "ngraph/op/matrix_nms.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void matrix_nms(const float* boxes_data,
                            const Shape& boxes_data_shape,
                            const float* scores_data,
                            const Shape& scores_data_shape,
                            op::util::NmsBase::SortResultType sort_result_type,
                            bool sort_result_across_batch,
                            float score_threshold,
                            int nms_top_k,
                            int keep_top_k,
                            int background_class,
                            const op::v8::MatrixNms::DecayFunction decay_function,
                            float gaussian_sigma,
                            float post_threshold,
                            bool normalized,
                            float* selected_outputs,
                            const Shape& selected_outputs_shape,
                            int64_t* selected_indices,
                            const Shape& selected_indices_shape,
                            int64_t* valid_outputs);

            void matrix_nms_postprocessing(const HostTensorVector& outputs,
                                           const ngraph::element::Type output_type,
                                           const std::vector<float>& selected_outputs,
                                           const std::vector<int64_t>& selected_indices,
                                           const std::vector<int64_t>& valid_outputs);
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
