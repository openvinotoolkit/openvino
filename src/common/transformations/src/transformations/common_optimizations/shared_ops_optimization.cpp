// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/shared_ops_optimization.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace {
#define ACCESSOR(type)                                                                \
    void on_adapter(const std::string& name, ValueAccessor<type>& adapter) override { \
        m_attributes_map.emplace_back(adapter.get());                                 \
    };
#define ACCESSOR_V(type) ACCESSOR(type) ACCESSOR(std::vector<type>)

class NodeComparingVisitor : public ov::AttributeVisitor {
public:
    NodeComparingVisitor(std::vector<Any>& attrs) : m_attributes_map(attrs) {}

    ACCESSOR(bool)
    ACCESSOR_V(std::string)
    ACCESSOR_V(int8_t)
    ACCESSOR_V(int16_t)
    ACCESSOR_V(int32_t)
    ACCESSOR_V(int64_t)
    ACCESSOR_V(uint8_t)
    ACCESSOR_V(uint16_t)
    ACCESSOR_V(uint32_t)
    ACCESSOR_V(uint64_t)
    ACCESSOR_V(float)
    ACCESSOR_V(double)

    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("Can not compare void");
    };
    void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("Can not compare void*");
    };
    void on_adapter(const std::string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("Can not compare models");
    };

    // we don't care about name of attributes since the Comparing is done on same type of nodes.
    std::string get_name_with_context() override {
        return {};
    }
    void start_structure(const std::string& name) override {}
    std::string finish_structure() override {
        return {};
    }

private:
    std::vector<Any>& m_attributes_map;
};

bool inputs_from_same_source_or_equal_constants(const std::shared_ptr<Node>& lhs,
                                                const std::shared_ptr<Node>& rhs,
                                                int shared_input_idx) {
    if (lhs->get_input_size() != rhs->get_input_size())
        return false;
    size_t input_size = lhs->get_input_size();
    for (size_t i = 0; i < input_size; ++i) {
        if (i == static_cast<size_t>(shared_input_idx))
            continue;
        if (lhs->input_value(i) == rhs->input_value(i))
            continue;
        auto lhs_constant = as_type_ptr<v0::Constant>(lhs->get_input_node_shared_ptr(i));
        auto rhs_constant = as_type_ptr<v0::Constant>(rhs->get_input_node_shared_ptr(i));
        if (!lhs_constant || !rhs_constant)
            return false;
        if (lhs_constant->get_element_type() != rhs_constant->get_element_type())
            return false;
        if (lhs_constant->get_shape() != rhs_constant->get_shape())
            return false;
        if (memcmp(lhs_constant->get_data_ptr(), rhs_constant->get_data_ptr(), lhs_constant->get_byte_size()) != 0)
            return false;
    }
    return true;
}

struct NodeInfo {
    std::shared_ptr<Node> node;
    std::vector<Any> attrs;
    bool is_attrs_extracted = false;
    bool is_attrs_valid = false;
    int shared_input_idx;
    NodeInfo(const std::shared_ptr<Node>& node, int shared_input_idx)
        : node(node),
          shared_input_idx(shared_input_idx) {}
    bool extract_attrs() {
        if (!is_attrs_extracted) {
            try {
                // lazy evaluate attr since it's slow
                NodeComparingVisitor attr_visitor(attrs);
                node->visit_attributes(attr_visitor);
                is_attrs_valid = true;
            } catch (...) {
                // we avoid errors during comparison of objects without equality operands
                // assuming they are not equal
                is_attrs_valid = false;
            }
            is_attrs_extracted = true;
        }
        return is_attrs_valid;
    }
    bool operator==(NodeInfo& other) {
        const auto& lhs = node;
        const auto& rhs = other.node;

        if (shared_input_idx != other.shared_input_idx)
            return false;

        // making sure that nodes are of the same type
        if (lhs->get_type_info() != rhs->get_type_info())
            return false;
        // skipping nodes with control dependencies since replacement may create a loop, and it is hard to detect
        // beforehand. currently control dependencies are used to order execution of Assign-ReadValue pairs. shared op
        // optimization is not relevant for them since one variable is connected to exactly one pair of Assign-ReadValue
        if (!lhs->get_control_dependents().empty() || !lhs->get_control_dependencies().empty())
            return false;
        if (!rhs->get_control_dependents().empty() || !rhs->get_control_dependencies().empty())
            return false;
        // skip comparing rt_info. example: fused_name may have different strings
        // compare attributes

        if (!extract_attrs()) {
            return false;
        }
        if (!other.extract_attrs()) {
            return false;
        }
        if (attrs != other.attrs) {
            return false;
        }
        return inputs_from_same_source_or_equal_constants(lhs, rhs, shared_input_idx);
    }
};

bool is_subgraph = false;

bool shared_node_optimization(const shared_ptr<Model>& model) {
    bool rewritten = false;
    std::unordered_map<std::shared_ptr<ov::Node>, size_t> index_map;
    const auto& order = model->get_ordered_ops();
    for (size_t i = 0; i < order.size(); ++i)
        index_map[order[i]] = i;

    int64_t total_compares = 0;
    int64_t total_visits = 0;
    int64_t total_output = 0;
    // int64_t total_hit = 0;

    for (const auto& op : order) {
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = ov::as_type_ptr<op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                is_subgraph = true;
                if (sub_graph)
                    rewritten = shared_node_optimization(sub_graph) || rewritten;
                is_subgraph = false;
            }
        }
        for (auto& output : op->outputs()) {
            const auto& target_inputs = output.get_target_inputs();
            if (target_inputs.size() <= 1)
                continue;  // nothing to optimize

            unordered_map<Node::type_info_t, vector<NodeInfo>> type_to_node;
            for (const auto& input : target_inputs) {
                if (input.get_index() != 0)
                    continue;
                if (auto node = input.get_node()->shared_from_this()) {
                    total_output++;
                    if (is_subgraph && index_map.count(node) == 0) {
                        if (node->get_type_name() != std::string("Constant"))
                            std::cout << node->get_type_name() << " == " << output << " => " << input << "\n";
                    }
                    type_to_node[node->get_type_info()].emplace_back(node, input.get_index());
                }
            }

            for (auto& item : type_to_node) {
                auto& shared_nodes = item.second;
                if (shared_nodes.size() < 2)
                    continue;
                // sort shared_nodes so that root would be the earliest in the topological order
                // it is critical for continuous application of this optimization
                std::sort(shared_nodes.begin(), shared_nodes.end(), [&index_map](const NodeInfo& a, const NodeInfo& b) {
                    return index_map[a.node] < index_map[b.node];
                });

                total_visits += shared_nodes.size();
                std::vector<bool> visited_nodes(shared_nodes.size(), false);
                for (size_t i = 0; i < visited_nodes.size(); ++i) {
                    if (visited_nodes[i])
                        continue;
                    const auto& root_op = shared_nodes[i].node;
                    visited_nodes[i] = true;

                    // no functionality is implemented to compare bodies of MultiSubGraphOp operations
                    if (ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(root_op)) {
                        continue;
                    }

                    for (size_t j = i + 1; j < visited_nodes.size(); ++j) {
                        if (visited_nodes[j])
                            continue;
                        const auto& child_op = shared_nodes[j].node;

                        if (shared_nodes[i] == shared_nodes[j]) {
                            rewritten =
                                replace_output_update_name(child_op->output(0), root_op->output(0)) || rewritten;
                            visited_nodes[j] = true;
                        }
                        total_compares++;
                    }
                }
            }
        }
    }
    return rewritten;
}

bool shape_of_upgrade(const shared_ptr<Model>& model) {
    bool rewritten = false;
    for (const auto& op : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = ov::as_type_ptr<op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                if (sub_graph)
                    rewritten = shape_of_upgrade(sub_graph) || rewritten;
            }
        } else if (auto v1_shape_of = ov::as_type_ptr<v0::ShapeOf>(op)) {
            auto v3_shape_of = std::make_shared<v3::ShapeOf>(v1_shape_of->input_value(0), element::i64);
            v3_shape_of->set_friendly_name(v1_shape_of->get_friendly_name());
            ov::replace_output_update_name(v1_shape_of, v3_shape_of);
            rewritten = true;
        }
    }
    return rewritten;
}

}  // namespace
bool pass::SharedOpOptimization::run_on_model(const shared_ptr<Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedOpOptimization);

    bool rewritten = shape_of_upgrade(model);
    rewritten = shared_node_optimization(model) || rewritten;
    return rewritten;
}
