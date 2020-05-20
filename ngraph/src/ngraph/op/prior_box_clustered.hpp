//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        struct NGRAPH_API PriorBoxClusteredAttrs
        {
            // widths         Desired widths of prior boxes
            // heights        Desired heights of prior boxes
            // clip           Clip output to [0,1]
            // step_widths    Distance between prior box centers
            // step_heights   Distance between prior box centers
            // offset         Box offset relative to top center of image
            // variances      Values to adjust prior boxes with
            std::vector<float> widths;
            std::vector<float> heights;
            bool clip = false;
            float step_widths = 1.0f;
            float step_heights = 1.0f;
            float offset = 0.0f;
            std::vector<float> variances;
        };

        namespace v0
        {
            /// \brief Layer which generates prior boxes of specified sizes
            /// normalized to input image size
            class NGRAPH_API PriorBoxClustered : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"PriorBoxClustered", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                PriorBoxClustered() = default;
                /// \brief Constructs a PriorBoxClustered operation
                ///
                /// \param layer_shape    Shape of layer for which prior boxes are computed
                /// \param image_shape    Shape of image to which prior boxes are scaled
                /// \param attrs          PriorBoxClustered attributes
                PriorBoxClustered(const Output<Node>& layer_shape,
                                  const Output<Node>& image_shape,
                                  const PriorBoxClusteredAttrs& attrs);

                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                const PriorBoxClusteredAttrs& get_attrs() const { return m_attrs; }
                virtual bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                PriorBoxClusteredAttrs m_attrs;
            };
        }
        using v0::PriorBoxClustered;
    }
}
