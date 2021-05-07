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

#include <cstdint>
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
            /// \brief An operation ExperimentalDetectronTopKROIs, according to the repository
            /// https://github.com/openvinotoolkit/training_extensions (see
            /// pytorch_toolkit/instance_segmentation/segmentoly/rcnn/roi_feature_extractor.py).
            class NGRAPH_API ExperimentalDetectronTopKROIs : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                ExperimentalDetectronTopKROIs() = default;
                /// \brief Constructs a ExperimentalDetectronTopKROIs operation.
                ///
                /// \param input_rois  Input rois
                /// \param rois_probs Probabilities for input rois
                /// \param max_rois Maximal numbers of output rois
                ExperimentalDetectronTopKROIs(const Output<Node>& input_rois,
                                              const Output<Node>& rois_probs,
                                              size_t max_rois = 0);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_max_rois() const { return m_max_rois; }

            private:
                size_t m_max_rois;
            };
        }
    }
}
