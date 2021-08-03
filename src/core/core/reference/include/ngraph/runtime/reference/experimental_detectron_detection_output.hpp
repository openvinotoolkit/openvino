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

#include <cmath>
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
            void experimental_detectron_detection_output(
                const float* input_rois,
                const float* input_deltas,
                const float* input_scores,
                const float* input_im_info,
                const op::v6::ExperimentalDetectronDetectionOutput::Attributes& attrs,
                float* output_boxes,
                float* output_scores,
                int32_t* output_classes);

            void experimental_detectron_detection_output_postprocessing(
                void* pboxes,
                void* pclasses,
                void* pscores,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_boxes,
                const std::vector<int32_t>& output_classes,
                const std::vector<float>& output_scores,
                const Shape& output_boxes_shape,
                const Shape& output_classes_shape,
                const Shape& output_scores_shape);
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
