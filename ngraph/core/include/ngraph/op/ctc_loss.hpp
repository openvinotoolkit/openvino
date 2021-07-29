// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            class NGRAPH_API CTCLoss : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"CTCLoss", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                CTCLoss() = default;
                /// \brief Constructs a CTCLoss operation
                ///
                /// \param logits                         3-D tensor of logits
                /// \param logit_length                   1-D tensor of length for each object from
                /// a batch
                /// \param labels                         2-D tensor of labels for which likelyhood
                /// is estimated using logist
                /// \param label_length                   1-D tensor of length for each label
                /// sequence
                /// \param blank_index                    Scalar used to mark a blank index
                /// \param preprocess_collapse_repeated   Flag for preprocessing labels before loss
                /// calculation
                /// \param ctc_merge_repeated             Flag for merging repeated characters in a
                /// potential alignment
                /// \param unique                         Flag to find unique elements in a target
                /// before matching with alignment
                CTCLoss(const Output<Node>& logits,
                        const Output<Node>& logit_length,
                        const Output<Node>& labels,
                        const Output<Node>& label_length,
                        const bool preprocess_collapse_repeated = false,
                        const bool ctc_merge_repeated = true,
                        const bool unique = false);

                CTCLoss(const Output<Node>& logits,
                        const Output<Node>& logit_length,
                        const Output<Node>& labels,
                        const Output<Node>& label_length,
                        const Output<Node>& blank_index,
                        const bool preprocess_collapse_repeated = false,
                        const bool ctc_merge_repeated = true,
                        const bool unique = false);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_preprocess_collapse_repeated() const
                {
                    return preprocess_collapse_repeated_;
                }
                bool get_ctc_merge_repeated() const { return ctc_merge_repeated_; }
                bool get_unique() const { return unique_; }

            private:
                bool preprocess_collapse_repeated_;
                bool ctc_merge_repeated_;
                bool unique_;
            };
        } // namespace v4
    }     // namespace op
} // namespace ngraph
