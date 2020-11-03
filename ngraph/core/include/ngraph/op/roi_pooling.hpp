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
                enum class ROIPoolingMethod
                {
                    Bilinear, // Bilinear interpolation
                    Max       // Maximum
                };

                ROIPooling() = default;
                /// \brief Constructs a ROIPooling operation
                ///
                /// \param input          Input feature map {N, C, H, W}
                /// \param coords         Coordinates of bounding boxes
                /// \param pooled_h       Height of the ROI output features
                /// \param pooled_w       Width of the ROI output features
                /// \param spatial_scale  Ratio of input feature map over input image size
                /// \param method         Method of pooling - Max or Bilinear
                ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const int pooled_h,
                           const int pooled_w,
                           const float spatial_scale,
                           const std::string& method = "max");

                ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const int pooled_h,
                           const int pooled_w,
                           const float spatial_scale,
                           ROIPoolingMethod method = ROIPoolingMethod::Max);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int get_pooled_h() const { return m_pooled_h; }
                int get_pooled_w() const { return m_pooled_w; }
                float get_spatial_scale() const { return m_spatial_scale; }
                ROIPoolingMethod get_method() const { return m_method; }
                bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                int m_pooled_h;
                int m_pooled_w;
                float m_spatial_scale;
                ROIPoolingMethod m_method;
            };
        } // namespace v0
        using v0::ROIPooling;
    } // namespace op

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::ROIPooling::ROIPoolingMethod& mode);

    template <>
    class NGRAPH_API AttributeAdapter<op::v0::ROIPooling::ROIPoolingMethod>
        : public EnumAttributeAdapterBase<op::v0::ROIPooling::ROIPoolingMethod>
    {
    public:
        AttributeAdapter(op::v0::ROIPooling::ROIPoolingMethod& value)
            : EnumAttributeAdapterBase<op::v0::ROIPooling::ROIPoolingMethod>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v0::ROIPooling::ROIPoolingMethod>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
}
