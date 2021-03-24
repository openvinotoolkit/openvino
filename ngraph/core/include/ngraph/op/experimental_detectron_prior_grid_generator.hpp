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
            /// \brief An operation ExperimentalDetectronPriorGridGenerator, according to
            /// the repository https://github.com/openvinotoolkit/training_extensions
            /// (see pytorch_toolkit/instance_segmentation/segmentoly/rcnn/prior_box.py).
            class NGRAPH_API ExperimentalDetectronPriorGridGenerator : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Structure that specifies attributes of the operation
                struct Attributes
                {
                    // Specifies whether the output tensor should be 2D or 4D
                    // `true`  means the output tensor should be 2D tensor,
                    // `false` means the output tensor should be 4D tensor.
                    bool flatten;
                    // Specifies number of cells of the generated grid with respect to height.
                    int64_t h;
                    // Specifies number of cells of the generated grid with respect to width.
                    int64_t w;
                    // Specifies the step of generated grid with respect to x coordinate
                    float stride_x;
                    // Specifies the step of generated grid with respect to y coordinate
                    float stride_y;
                };

                ExperimentalDetectronPriorGridGenerator() = default;
                /// \brief Constructs a ExperimentalDetectronDetectionOutput operation.
                ///
                /// \param priors  Input priors
                /// \param feature_map Input feature map
                /// \param im_data Image data
                /// \param attrs   attributes
                ExperimentalDetectronPriorGridGenerator(const Output<Node>& priors,
                                                        const Output<Node>& feature_map,
                                                        const Output<Node>& im_data,
                                                        const Attributes& attrs);
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \brief Returns attributes of this operation.
                const Attributes& get_attrs() const { return m_attrs; }

            private:
                Attributes m_attrs;

                void validate();
            };
        }
    }
}
