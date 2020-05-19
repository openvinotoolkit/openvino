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

#include <unordered_map>

#include "ngraph/code_writer.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class CommonFunctionCollection;
    }
}

class NGRAPH_API ngraph::pass::CommonFunctionCollection : public ModulePass
{
public:
    /// \brief Create the CommonFunctionCollection pass
    /// \param function_emitter - This is a function that takes a reference to a Node and as string.
    ///        The string is the name of the emitted function and the body of the function is
    ///        the code for the op.
    /// \param result_map - This is a mapping of source node -> emitted static function node, where
    ///       the key is the source node and the value is the emitted static function node. The
    ///        name of the function to call is create_function_name(<emitted static function node>)
    /// \param emitted_functions - string to contain the emitted code for all of the static
    ///        functions.
    CommonFunctionCollection(std::function<std::string(Node&, std::string)> function_emitter,
                             std::unordered_map<Node*, Node*>& result_map,
                             std::string& emitted_functions);

    virtual ~CommonFunctionCollection() override;

    bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) override;

    /// \brief Construct the name of the function to call for this op
    /// \param node - Node used to construct the function name. This node is the `value` of the
    ///        result_map passed to the pass's constructor.
    /// \return string containing the name of the function to be called
    static std::string create_function_name(const Node& node);

private:
    std::function<std::string(Node&, std::string)> m_emit_op_as_function;
    std::unordered_map<Node*, Node*>& m_node_function_map;
    std::string& m_emitted_functions;
};
