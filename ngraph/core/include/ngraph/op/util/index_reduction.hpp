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

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            class NGRAPH_API IndexReduction : public Op
            {
            protected:
                IndexReduction();

                IndexReduction(const Output<Node>& arg,
                               uint64_t axis,
                               const element::Type& index_element_type);

            public:
                uint64_t get_reduction_axis() const;
                void set_reduction_axis(uint64_t value);
                element::Type get_index_element_type() const;
                void set_index_element_type(const element::Type& index_element_type);
                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

            protected:
                uint64_t m_axis{0};
                element::Type m_index_element_type;
            };
        }
    }
}
