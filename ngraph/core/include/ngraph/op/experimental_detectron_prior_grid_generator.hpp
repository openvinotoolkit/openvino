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
            /// \brief An operation ExperimentalDetectronPriorGridGenerator generates prior
            /// grids of specified sizes.
            class NGRAPH_API ExperimentalDetectronPriorGridGenerator : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Structure that specifies attributes of the operation
                struct Attributes
                {
                    // Specifies whether the output tensor should be 2D or 4D
                    // `true`  means the output tensor should be 2D tensor,
                    // `false` means the output tensor should be 4D tensor.
                    bool flatten;
                    // Specifies number of cells of the generated grid with respect to height.
                    int64_t h;
                    // Specifies number of cells of the generated grid with respect to width.
                    int64_t w;
                    // Specifies the step of generated grid with respect to x coordinate
                    float stride_x;
                    // Specifies the step of generated grid with respect to y coordinate
                    float stride_y;
                };

                ExperimentalDetectronPriorGridGenerator() = default;
                /// \brief Constructs a ExperimentalDetectronDetectionOutput operation.
                ///
                /// \param priors  Input priors
                /// \param feature_map Input feature map
                /// \param im_data Image data
                /// \param attrs   attributes
                ExperimentalDetectronPriorGridGenerator(const Output<Node>& priors,
                                                        const Output<Node>& feature_map,
                                                        const Output<Node>& im_data,
                                                        const Attributes& attrs);
                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \brief Returns attributes of this operation.
                const Attributes& get_attrs() const { return m_attrs; }

            private:
                Attributes m_attrs;

                void validate();
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
