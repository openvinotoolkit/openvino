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

#include <utility>

#include "ngraph/op/passthrough.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Passthrough::type_info;

ngraph::op::Passthrough::Passthrough(const std::string& logical_type,
                                     const std::string& language,
                                     const std::string& function,
                                     const OutputVector& args,
                                     std::vector<std::tuple<element::Type, PartialShape>> outputs)
    : Op{args}
    , m_logical_type{logical_type}
    , m_language{language}
    , m_function{function}
    , m_output_shapes{std::move(outputs)}
{
    set_output_size(m_output_shapes.size());
    constructor_validate_and_infer_types();
}

void ngraph::op::Passthrough::validate_and_infer_types()
{
    // N.B. It would be useful to have the backend deduce the output
    //      shapes, instead of having them passed in via the
    //      constructor and trusting that they're correct.
    //
    //      The primary barrier to doing so is that at the point where
    //      Passthrough ops are being constructed, we don't
    //      necessarily have the backend available.
    //
    //      At some point, we may want to add higher-level
    //      backend-specific APIs for constructing Passthrough
    //      operations; that would ensure that the backend can
    //      understand the language being used, and would allow the
    //      backend to infer the output shapes as needed.

    std::size_t idx = 0;
    for (auto& output_shape : m_output_shapes)
    {
        set_output_type(idx++, std::get<0>(output_shape), std::get<1>(output_shape));
    }
}

std::shared_ptr<ngraph::Node>
    ngraph::op::Passthrough::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != get_input_size())
    {
        throw ngraph_error{
            "Passthrough node input counts cannot be changed for a given Passthrough function"};
    }
    return std::make_shared<Passthrough>(
        m_logical_type, m_language, m_function, new_args, m_output_shapes);
}
