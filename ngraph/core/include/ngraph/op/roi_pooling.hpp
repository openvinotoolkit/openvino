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
        namespace v0
        {
            class NGRAPH_API ROIPooling : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ROIPooling", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ROIPooling() = default;
                /// \brief Constructs a ROIPooling operation
                ///
                /// \param input          Input feature map {N, C, ...}
                /// \param coords         Coordinates of bounding boxes
                /// \param output_size    Height/Width of ROI output features
                /// \param spatial_scale  Ratio of input feature map over input image size
                /// \param method         Method of pooling - Max or Bilinear
                ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const Shape& output_size,
                           const float spatial_scale,
                           const std::string& method);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Shape& get_output_size() const { return m_output_size; }
                float get_spatial_scale() const { return m_spatial_scale; }
                const std::string& get_method() const { return m_method; }
                bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                Shape m_output_size;
                float m_spatial_scale;
                std::string m_method;
            };
        }
        using v0::ROIPooling;
    }
}
