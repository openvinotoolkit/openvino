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
        namespace v0
        {
            class NGRAPH_API RegionYolo : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"RegionYolo", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                RegionYolo() = default;
                ///
                /// \brief      Constructs a RegionYolo operation
                ///
                /// \param[in]  input        Input
                /// \param[in]  coords       Number of coordinates for each region
                /// \param[in]  classes      Number of classes for each region
                /// \param[in]  regions      Number of regions
                /// \param[in]  do_softmax   Compute softmax
                /// \param[in]  mask         Mask
                /// \param[in]  axis         Axis to begin softmax on
                /// \param[in]  end_axis     Axis to end softmax on
                /// \param[in]  anchors      A flattened list of pairs `[width, height]` that
                /// describes
                ///                          prior box sizes.
                ///
                RegionYolo(const Output<Node>& input,
                           const size_t coords,
                           const size_t classes,
                           const size_t regions,
                           const bool do_softmax,
                           const std::vector<int64_t>& mask,
                           const int axis,
                           const int end_axis,
                           const std::vector<float>& anchors = std::vector<float>{});

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_num_coords() const { return m_num_coords; }
                size_t get_num_classes() const { return m_num_classes; }
                size_t get_num_regions() const { return m_num_regions; }
                bool get_do_softmax() const { return m_do_softmax; }
                const std::vector<int64_t>& get_mask() const { return m_mask; }
                const std::vector<float>& get_anchors() const { return m_anchors; }
                int get_axis() const { return m_axis; }
                int get_end_axis() const { return m_end_axis; }

            private:
                size_t m_num_coords;
                size_t m_num_classes;
                size_t m_num_regions;
                bool m_do_softmax;
                std::vector<int64_t> m_mask;
                std::vector<float> m_anchors{};
                int m_axis;
                int m_end_axis;
            };
        } // namespace v0
        using v0::RegionYolo;
    } // namespace op
} // namespace ngraph
