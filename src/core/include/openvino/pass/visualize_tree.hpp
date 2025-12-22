// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "openvino/pass/pass.hpp"

class HeightMap;

using visualize_tree_ops_map_t =
    std::unordered_map<ov::Node::type_info_t, std::function<void(const ov::Node&, std::ostream& ss)>>;

namespace ov::pass {
/**
 * @brief VisualizeTree pass allows to serialize ov::Model to xDot format
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API VisualizeTree : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::pass::VisualizeTree");

    using node_modifiers_t = std::function<void(const Node& node, std::vector<std::string>& attributes)>;

    VisualizeTree(const std::string& file_name, node_modifiers_t nm = nullptr, bool dot_only = false);
    VisualizeTree(std::filesystem::path file_name, node_modifiers_t nm = nullptr, bool dot_only = false);

    template <class TPath>
    VisualizeTree(const TPath& file_name, node_modifiers_t nm = nullptr, bool dot_only = false)
        : VisualizeTree(
              [&] {
                  static_assert(std::is_constructible_v<std::filesystem::path, std::decay_t<TPath>>,
                                "file_name type is not supported");
                  return std::string(file_name);
              }(),
              nm,
              dot_only) {}

    bool run_on_model(const std::shared_ptr<ov::Model>&) override;

    void set_ops_to_details(const visualize_tree_ops_map_t& ops_map) {
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
    std::filesystem::path m_name;
    std::set<std::shared_ptr<Node>> m_nodes_with_attributes;
    visualize_tree_ops_map_t m_ops_to_details;
    node_modifiers_t m_node_modifiers = nullptr;
    bool m_dot_only;
    static constexpr int max_jump_distance = 20;
    std::unordered_map<std::shared_ptr<ov::Symbol>, size_t> m_symbol_to_name;
};
}  // namespace ov::pass
