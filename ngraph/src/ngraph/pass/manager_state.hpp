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

#include <functional>
#include <initializer_list>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"

using visualize_tree_ops_map_t =
    std::unordered_map<ngraph::Node::type_info_t,
                       std::function<void(const ngraph::Node&, std::ostream& ss)>>;

namespace ngraph
{
    namespace pass
    {
        class ManagerState;
    }
}

class ngraph::pass::ManagerState
{
public:
    void set_visualize_tree_ops_map(const visualize_tree_ops_map_t& ops_map)
    {
        m_visualize_tree_ops_map = ops_map;
    }

    const visualize_tree_ops_map_t& get_visualize_tree_ops_map()
    {
        return m_visualize_tree_ops_map;
    }

    void set_function(const std::shared_ptr<Function> function) { m_function = function; }
    std::shared_ptr<Function> get_function() const { return m_function; }
    std::vector<std::shared_ptr<Function>> get_functions() const
        NGRAPH_DEPRECATED("Use get_function()")
    {
        return {m_function};
    }

private:
    visualize_tree_ops_map_t m_visualize_tree_ops_map;
    std::shared_ptr<Function> m_function;
};
