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
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "ngraph/pass/manager_state.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class VisualizeTree;
    }
}

class HeightMap;

class NGRAPH_API ngraph::pass::VisualizeTree : public ModulePass
{
public:
    using node_modifiers_t =
        std::function<void(const Node& node, std::vector<std::string>& attributes)>;
    VisualizeTree(const std::string& file_name,
                  node_modifiers_t nm = nullptr,
                  bool dot_only = false);
    bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) override;

    void set_ops_to_details(const visualize_tree_ops_map_t& ops_map) { m_ops_to_details = ops_map; }
protected:
    void add_node_arguments(std::shared_ptr<Node> node,
                            std::unordered_map<Node*, HeightMap>& height_maps,
                            size_t& fake_node_ctr);
    std::string add_attributes(std::shared_ptr<Node> node);
    virtual std::string get_attributes(std::shared_ptr<Node> node);
    virtual std::string get_node_name(std::shared_ptr<Node> node);
    void render() const;

    std::stringstream m_ss;
    std::string m_name;
    std::set<std::shared_ptr<Node>> m_nodes_with_attributes;
    visualize_tree_ops_map_t m_ops_to_details;
    node_modifiers_t m_node_modifiers = nullptr;
    bool m_dot_only;
    static const int max_jump_distance;
};
