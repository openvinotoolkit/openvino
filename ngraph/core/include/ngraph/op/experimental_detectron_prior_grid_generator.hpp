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
            class NGRAPH_API ExperimentalDetectronPriorGridGenerator : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                struct Attributes
                {
                    bool flatten;
                    int h;
                    int w;
                    float stride_x;
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

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Attributes& get_attrs() const { return m_attrs; }
            private:
                Attributes m_attrs;
            };
        } // namespace v6
        using v6::ExperimentalDetectronPriorGridGenerator;
    } // namespace op
} // namespace ngraph
