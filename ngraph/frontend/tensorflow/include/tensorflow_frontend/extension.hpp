//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <functional>

#include <frontend_manager/frontend.hpp>
#include "node_context.hpp"

namespace ngraph
{
namespace frontend
{
/// \brief Builds ngraph fragment for a specific operation in original TF graph
/** An extension is a pair of a string and a function; the string specifies type of the op
    as it is define in Tneand the function builds ngraph function fragment based on NodeContext. */
class TFConversionExtension : public Extension
{
public:

    // Signature for a function that build ngraph fragment by a TF op
    typedef std::function<ngraph::OutputVector(const ngraph::frontend::tensorflow::detail::NodeContext&)> Converter;

    TFConversionExtension (const std::string& op_type, const Converter& converter) :
        m_op_type(op_type),
        m_converter(converter)
    {}

    const std::string& get_op_type () const { return m_op_type; }
    const Converter& get_converter () const { return m_converter; }

private:

    std::string m_op_type;
    Converter m_converter;
};

} // namespace frontend
} // namespace ngraph
