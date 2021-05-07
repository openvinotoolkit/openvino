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
            /// \brief An operation ExperimentalDetectronGenerateProposalsSingleImage, according to
            /// the repository https://github.com/openvinotoolkit/training_extensions
            /// (see pytorch_toolkit/instance_segmentation/segmentoly/rcnn/proposal.py).
            class NGRAPH_API ExperimentalDetectronGenerateProposalsSingleImage : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Structure that specifies attributes of the operation
                struct Attributes
                {
                    // minimum box width & height
                    float min_size;
                    // specifies NMS threshold
                    float nms_threshold;
                    // number of top-n proposals after NMS
                    int64_t post_nms_count;
                    // number of top-n proposals before NMS
                    int64_t pre_nms_count;
                };

                ExperimentalDetectronGenerateProposalsSingleImage() = default;
                /// \brief Constructs a ExperimentalDetectronGenerateProposalsSingleImage operation.
                ///
                /// \param im_info Input image info
                /// \param anchors Input anchors
                /// \param deltas Input deltas
                /// \param scores Input scores
                /// \param attrs Operation attributes
                ExperimentalDetectronGenerateProposalsSingleImage(const Output<Node>& im_info,
                                                                  const Output<Node>& anchors,
                                                                  const Output<Node>& deltas,
                                                                  const Output<Node>& scores,
                                                                  const Attributes& attrs);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Attributes& get_attrs() const { return m_attrs; }

            private:
                Attributes m_attrs;
            };
        }
    }
}
