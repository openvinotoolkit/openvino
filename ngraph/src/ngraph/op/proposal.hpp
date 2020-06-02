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
        // base_size       Anchor sizes
        // pre_nms_topn    Number of boxes before nms
        // post_nms_topn   Number of boxes after nms
        // nms_thresh      Threshold for nms
        // feat_stride     Feature stride
        // min_size        Minimum box size
        // ratio   Ratios for anchor generation
        // scale   Scales for anchor generation
        // clip_before_nms Clip before NMs
        // clip_after_nms  Clip after NMs
        // normalize       Normalize boxes to [0,1]
        // box_size_scale  Scale factor for scaling box size logits
        // box_coordinate_scale Scale factor for scaling box coordiate logits
        // framework            Calculation frameworkrithm to use
        struct ProposalAttrs
        {
            size_t base_size;
            size_t pre_nms_topn;
            size_t post_nms_topn;
            float nms_thresh = 0.0f;
            size_t feat_stride = 1;
            size_t min_size = 1;
            std::vector<float> ratio;
            std::vector<float> scale;
            bool clip_before_nms = false;
            bool clip_after_nms = false;
            bool normalize = false;
            float box_size_scale = 1.0f;
            float box_coordinate_scale = 1.0f;
            std::string framework;
        };

        namespace v0
        {
            class NGRAPH_API Proposal : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Proposal", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Proposal() = default;
                /// \brief Constructs a Proposal operation
                ///
                /// \param class_probs     Class probability scores
                /// \param class_logits    Class prediction logits
                /// \param image_shape     Shape of image
                /// \param attrs           Proposal op attributes
                Proposal(const Output<Node>& class_probs,
                         const Output<Node>& class_logits,
                         const Output<Node>& image_shape,
                         const ProposalAttrs& attrs);

                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                const ProposalAttrs& get_attrs() const { return m_attrs; }
                virtual bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                ProposalAttrs m_attrs;
            };
        }
        using v0::Proposal;
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::ProposalAttrs> : public VisitorAdapter
    {
    public:
        AttributeAdapter(op::ProposalAttrs& ref);

        virtual bool visit_attributes(AttributeVisitor& visitor) override;
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::ProposalAttrs>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    protected:
        op::ProposalAttrs& m_ref;
    };
}
