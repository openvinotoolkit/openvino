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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class NGRAPH_API DeformablePSROIPooling : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"DeformablePSROIPooling", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                DeformablePSROIPooling() = default;
                /// \brief Constructs a DeformablePSROIPooling operation
                ///
                /// \param input           Input tensor with feature maps
                /// \param coords          Input tensor describing box consisting
                ///                        of five element tuples
                /// \param offsets         Input blob with transformation values
                /// \param output_dim      Pooled output channel number
                /// \param group_size      Number of groups to encode position-sensitive score maps
                /// \param spatial_scale   Multiplicative spatial scale factor to translate ROI
                ///                        coordinates from their input scale to the scale used when
                ///                        pooling
                /// \param mode            Specifies mode for pooling.
                /// \param spatial_bins_x  Specifies numbers of bins to divide the input feature
                ///                         maps over width
                /// \param spatial_bins_y  Specifies numbers of bins to divide the input feature
                ///                        maps over height
                /// \param no_trans        The flag that specifies whenever third input exists
                ///                        and contains transformation (offset) values
                /// \param trans_std       The value that all transformation (offset) values are
                ///                        multiplied with
                /// \param part_size       The number of parts the output tensor spatial dimensions
                ///                        are divided into. Basically it is the height
                ///                        and width of the third input
                DeformablePSROIPooling(const Output<Node>& input,
                                       const Output<Node>& coords,
                                       const Output<Node>& offsets,
                                       const int64_t output_dim,
                                       const float spatial_scale,
                                       const int64_t group_size = 1,
                                       const std::string mode = "bilinear_deformable",
                                       int64_t spatial_bins_x = 1,
                                       int64_t spatial_bins_y = 1,
                                       float trans_std = 1,
                                       int64_t part_size = 1);

                DeformablePSROIPooling(const Output<Node>& input,
                                       const Output<Node>& coords,
                                       const int64_t output_dim,
                                       const float spatial_scale,
                                       const int64_t group_size = 1,
                                       const std::string mode = "bilinear_deformable",
                                       int64_t spatial_bins_x = 1,
                                       int64_t spatial_bins_y = 1,
                                       float trans_std = 1,
                                       int64_t part_size = 1);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int64_t get_output_dim() const { return m_output_dim; }
                int64_t get_group_size() const { return m_group_size; }
                float get_spatial_scale() const { return m_spatial_scale; }
                const std::string& get_mode() const { return m_mode; }
                int64_t get_spatial_bins_x() const { return m_spatial_bins_x; }
                int64_t get_spatial_bins_y() const { return m_spatial_bins_y; }
                float get_trans_std() const { return m_trans_std; }
                int64_t get_part_size() const { return m_part_size; }

            private:
                int64_t m_output_dim;
                float m_spatial_scale;
                int64_t m_group_size = 1;
                std::string m_mode = "bilinear";
                int64_t m_spatial_bins_x = 1;
                int64_t m_spatial_bins_y = 1;
                float m_trans_std = 1.f;
                int64_t m_part_size = 1;
            };
        }
    }
}
