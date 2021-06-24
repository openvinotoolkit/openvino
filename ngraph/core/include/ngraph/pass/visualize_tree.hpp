// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "ngraph/pass/pass.hpp"

class HeightMap;

using visualize_tree_ops_map_t =
    std::unordered_map<ngraph::Node::type_info_t,
                       std::function<void(const ngraph::Node&, std::ostream& ss)>>;

namespace ngraph
{
    namespace pass
    {
        class NGRAPH_API VisualizeTree : public FunctionPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            using node_modifiers_t =
                std::function<void(const Node& node, std::vector<std::string>& attributes)>;
            VisualizeTree(const std::string& file_name,
                          node_modifiers_t nm = nullptr,
                          bool dot_only = false);
            bool run_on_function(std::shared_ptr<ngraph::Function>) override;

            void set_ops_to_details(const visualize_tree_ops_map_t& ops_map)
            {
                m_ops_to_details = ops_map;
            }

        protected:
            void add_node_arguments(std::shared_ptr<Node> node,
                                    std::unordered_map<Node*, HeightMap>& height_maps,
                                    size_t& fake_node_ctr);
            std::string add_attributes(std::shared_ptr<Node> node);
            virtual std::string get_attributes(std::shared_ptr<Node> node);
            virtual std::string get_node_name(std::shared_ptr<Node> node);
            std::string get_constant_value(std::shared_ptr<Node> node, size_t max_elements = 7);

            void render() const;

            std::stringstream m_ss;
            std::string m_name;
            std::set<std::shared_ptr<Node>> m_nodes_with_attributes;
            visualize_tree_ops_map_t m_ops_to_details;
            node_modifiers_t m_node_modifiers = nullptr;
            bool m_dot_only;
            static const int max_jump_distance;
        };
    } // namespace pass
} // namespace ngraph
