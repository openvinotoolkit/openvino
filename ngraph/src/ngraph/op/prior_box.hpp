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
        struct PriorBoxAttrs
        {
            // min_size         Desired min_size of prior boxes
            // max_size         Desired max_size of prior boxes
            // aspect_ratio     Aspect ratios of prior boxes
            // clip             Clip output to [0,1]
            // flip             Flip aspect ratios
            // step             Distance between prior box centers
            // offset           Box offset relative to top center of image
            // variance         Values to adjust prior boxes with
            // scale_all_sizes  Scale all sizes
            std::vector<float> min_size;
            std::vector<float> max_size;
            std::vector<float> aspect_ratio;
            std::vector<float> density;
            std::vector<float> fixed_ratio;
            std::vector<float> fixed_size;
            bool clip = false;
            bool flip = false;
            float step = 1.0f;
            float offset = 0.0f;
            std::vector<float> variance;
            bool scale_all_sizes = false;
        };

        namespace v0
        {
            /// \brief Layer which generates prior boxes of specified sizes
            /// normalized to input image size
            class NGRAPH_API PriorBox : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"PriorBox", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                PriorBox() = default;
                /// \brief Constructs a PriorBox operation
                ///
                /// \param layer_shape    Shape of layer for which prior boxes are computed
                /// \param image_shape    Shape of image to which prior boxes are scaled
                /// \param attrs          PriorBox attributes
                PriorBox(const Output<Node>& layer_shape,
                         const Output<Node>& image_shape,
                         const PriorBoxAttrs& attrs);

                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                static int64_t number_of_priors(const PriorBoxAttrs& attrs);

                static std::vector<float>
                    normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip);
                const PriorBoxAttrs& get_attrs() const { return m_attrs; }
                virtual bool visit_attributes(AttributeVisitor& visitor) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            private:
                PriorBoxAttrs m_attrs;
            };
        }
        using v0::PriorBox;
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::PriorBoxAttrs> : public VisitorAdapter
    {
    public:
        AttributeAdapter(op::PriorBoxAttrs& ref);

        virtual bool visit_attributes(AttributeVisitor& visitor) override;
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::PriorBoxAttrs>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    protected:
        op::PriorBoxAttrs& m_ref;
    };
}
