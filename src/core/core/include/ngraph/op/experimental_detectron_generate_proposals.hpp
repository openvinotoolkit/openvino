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
            /// \brief An operation ExperimentalDetectronGenerateProposalsSingleImage
            /// computes ROIs and their scores based on input data.
            class NGRAPH_API ExperimentalDetectronGenerateProposalsSingleImage : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Structure that specifies attributes of the operation
                struct Attributes
                {
                    // minimum box width & height
                    float min_size;
                    // specifies NMS threshold
                    float nms_threshold;
                    // number of top-n proposals after NMS
                    int64_t post_nms_count;
                    // number of top-n proposals before NMS
                    int64_t pre_nms_count;
                };

                ExperimentalDetectronGenerateProposalsSingleImage() = default;
                /// \brief Constructs a ExperimentalDetectronGenerateProposalsSingleImage operation.
                ///
                /// \param im_info Input image info
                /// \param anchors Input anchors
                /// \param deltas Input deltas
                /// \param scores Input scores
                /// \param attrs Operation attributes
                ExperimentalDetectronGenerateProposalsSingleImage(const Output<Node>& im_info,
                                                                  const Output<Node>& anchors,
                                                                  const Output<Node>& deltas,
                                                                  const Output<Node>& scores,
                                                                  const Attributes& attrs);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Attributes& get_attrs() const { return m_attrs; }

            private:
                Attributes m_attrs;
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
