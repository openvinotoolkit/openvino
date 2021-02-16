//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstddef>
#include <cstdint>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void experimental_detectron_prior_grid_generator(const float* priors,
                                                             const Shape& priors_shape,
                                                             const Shape& feature_map_shape,
                                                             const Shape& im_data_shape,
                                                             float* output_rois,
                                                             int64_t grid_h,
                                                             int64_t grid_w,
                                                             float stride_h,
                                                             float stride_w);

            void experimental_detectron_prior_grid_generator_postprocessing(
                const HostTensorVector& outputs,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_rois,
                const Shape& output_rois_shape);
        }
    }
}
