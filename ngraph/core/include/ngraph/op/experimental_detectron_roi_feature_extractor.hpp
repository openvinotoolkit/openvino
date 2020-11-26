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
            class NGRAPH_API ExperimentalDetectronROIFeatureExtractor : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                struct Attributes
                {
                    int output_dim = 0;
                    std::vector<int> pyramid_scales;
                    int sampling_ratio = 0;
                    bool aligned = false;
                };

                ExperimentalDetectronROIFeatureExtractor() = default;
                /// \brief Constructs a ExperimentalDetectronROIFeatureExtractor operation.
                ///
                /// \param args  The outputs producing the input tensors.
                /// \param attrs  Attributes attributes
                ExperimentalDetectronROIFeatureExtractor(const OutputVector& args,
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
        using v6::ExperimentalDetectronROIFeatureExtractor;
    } // namespace op
} // namespace ngraph
