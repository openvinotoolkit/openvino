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
#include <vector>
#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v6
        {
            /// \brief An operation ExperimentalDetectronDetectionOutput, according to
            /// the repository https://github.com/openvinotoolkit/training_extensions
            /// (see pytorch_toolkit/instance_segmentation/segmentoly/rcnn/detection_output.py).
            class NGRAPH_API ExperimentalDetectronDetectionOutput : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Structure that specifies attributes of the operation
                struct Attributes
                {
                    // specifies score threshold
                    float score_threshold;
                    // specifies NMS threshold
                    float nms_threshold;
                    // specifies maximal delta of logarithms for width and height
                    float max_delta_log_wh;
                    // specifies number of detected classes
                    int64_t num_classes;
                    // specifies maximal number of detections per class
                    int64_t post_nms_count;
                    // specifies maximual number of detections per image
                    size_t max_detections_per_image;
                    // a flag specifies whether to delete background classes or not
                    // `true`  means background classes should be deleted,
                    // `false` means background classes shouldn't be deleted.
                    bool class_agnostic_box_regression;
                    // specifies deltas of weights
                    std::vector<float> deltas_weights;
                };

                ExperimentalDetectronDetectionOutput() = default;
                /// \brief Constructs a ExperimentalDetectronDetectionOutput operation.
                ///
                /// \param input_rois  Input rois
                /// \param input_deltas Input deltas
                /// \param input_scores Input scores
                /// \param input_im_info Input image info
                /// \param attrs  Attributes attributes
                ExperimentalDetectronDetectionOutput(const Output<Node>& input_rois,
                                                     const Output<Node>& input_deltas,
                                                     const Output<Node>& input_scores,
                                                     const Output<Node>& input_im_info,
                                                     const Attributes& attrs);
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \brief Returns attributes of the operation ExperimentalDetectronDetectionOutput
                const Attributes& get_attrs() const { return m_attrs; }

            private:
                Attributes m_attrs;
            };
        }
    }
}
