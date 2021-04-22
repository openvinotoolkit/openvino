// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>
#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v6
        {
            /// \brief An operation ExperimentalDetectronTopKROIs, according to the repository
            /// is TopK operation applied to probabilities of input ROIs.
            class NGRAPH_API ExperimentalDetectronTopKROIs : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                ExperimentalDetectronTopKROIs() = default;
                /// \brief Constructs a ExperimentalDetectronTopKROIs operation.
                ///
                /// \param input_rois  Input rois
                /// \param rois_probs Probabilities for input rois
                /// \param max_rois Maximal numbers of output rois
                ExperimentalDetectronTopKROIs(const Output<Node>& input_rois,
                                              const Output<Node>& rois_probs,
                                              size_t max_rois = 0);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_max_rois() const { return m_max_rois; }

            private:
                size_t m_max_rois;
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
