// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
            bool clip = true;
            float step_widths = 0.0f;
            float step_heights = 0.0f;
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
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                PriorBoxClusteredAttrs m_attrs;
            };
        } // namespace v0
        using v0::PriorBoxClustered;
    } // namespace op
} // namespace ngraph
