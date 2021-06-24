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
            class NGRAPH_API PSROIPooling : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"PSROIPooling", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                PSROIPooling() = default;
                /// \brief Constructs a PSROIPooling operation
                ///
                /// \param input          Input feature map {N, C, ...}
                /// \param coords         Coordinates of bounding boxes
                /// \param output_dim     Output channel number
                /// \param group_size     Number of groups to encode position-sensitive scores
                /// \param spatial_scale  Ratio of input feature map over input image size
                /// \param spatial_bins_x Numbers of bins to divide the input feature maps over
                /// width
                /// \param spatial_bins_y Numbers of bins to divide the input feature maps over
                /// height
                /// \param mode           Mode of pooling - Avg or Bilinear
                PSROIPooling(const Output<Node>& input,
                             const Output<Node>& coords,
                             const size_t output_dim,
                             const size_t group_size,
                             const float spatial_scale,
                             int spatial_bins_x,
                             int spatial_bins_y,
                             const std::string& mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_output_dim() const { return m_output_dim; }
                size_t get_group_size() const { return m_group_size; }
                float get_spatial_scale() const { return m_spatial_scale; }
                int get_spatial_bins_x() const { return m_spatial_bins_x; }
                int get_spatial_bins_y() const { return m_spatial_bins_y; }
                const std::string& get_mode() const { return m_mode; }

            private:
                size_t m_output_dim;
                size_t m_group_size;
                float m_spatial_scale;
                int m_spatial_bins_x;
                int m_spatial_bins_y;
                std::string m_mode;
            };
        } // namespace v0
        using v0::PSROIPooling;
    } // namespace op
} // namespace ngraph
