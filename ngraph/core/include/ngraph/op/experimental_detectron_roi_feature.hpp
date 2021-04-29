// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
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
            /// \brief An operation ExperimentalDetectronROIFeatureExtractor
            /// is the ROIAlign operation applied over a feature pyramid.
            class NGRAPH_API ExperimentalDetectronROIFeatureExtractor : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Structure that specifies attributes of the operation
                struct Attributes
                {
                    int64_t output_size;
                    int64_t sampling_ratio;
                    std::vector<int64_t> pyramid_scales;
                    bool aligned;
                };

                ExperimentalDetectronROIFeatureExtractor() = default;
                /// \brief Constructs a ExperimentalDetectronROIFeatureExtractor operation.
                ///
                /// \param args  Inputs of ExperimentalDetectronROIFeatureExtractor
                /// \param attrs  Operation attributes
                ExperimentalDetectronROIFeatureExtractor(const OutputVector& args,
                                                         const Attributes& attrs);

                /// \brief Constructs a ExperimentalDetectronROIFeatureExtractor operation.
                ///
                /// \param args  Inputs of ExperimentalDetectronROIFeatureExtractor
                /// \param attrs  Operation attributes
                ExperimentalDetectronROIFeatureExtractor(const NodeVector& args,
                                                         const Attributes& attrs);
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \brief Returns attributes of the operation.
                const Attributes& get_attrs() const { return m_attrs; }

            private:
                Attributes m_attrs;
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
