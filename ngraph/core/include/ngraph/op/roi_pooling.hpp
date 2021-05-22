// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
                /// \param input          Input feature map {N, C, H, W}
                /// \param coords         Coordinates of bounding boxes
                /// \param output_size    Height/Width of ROI output features
                /// \param spatial_scale  Ratio of input feature map over input image size
                /// \param method         Method of pooling - Max or Bilinear
                ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const Shape& output_size,
                           const float spatial_scale,
                           const std::string& method = "max");

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Shape& get_output_size() const { return m_output_size; }
                float get_spatial_scale() const { return m_spatial_scale; }
                const std::string& get_method() const { return m_method; }
                bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                Shape m_output_size{0, 0};
                float m_spatial_scale;
                std::string m_method = "max";
            };

        } // namespace v0
        using v0::ROIPooling;

    } // namespace op

} // namespace ngraph
