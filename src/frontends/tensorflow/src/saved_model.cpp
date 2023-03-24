// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <fstream>
#include <string>

#include "graph_iterator_saved_model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "tensor_bundle.pb.h"
#include "trackable_object_graph.pb.h"

#ifdef ENABLE_SNAPPY_COMPRESSION
#    include "snappy.h"
#endif

namespace ov {
namespace frontend {
namespace tensorflow {

bool GraphIteratorSavedModel::is_valid_signature(const ::tensorflow::SignatureDef& signature) const {
    const std::map<::tensorflow::DataType, ov::element::Type> types{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16},
        {::tensorflow::DataType::DT_STRING, ov::element::undefined}};

    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    return true;
}

struct PtrNode {
    const ::tensorflow::NodeDef* node;
    std::vector<PtrNode*> inputs;
    std::vector<PtrNode*> outputs;

    PtrNode() : node(nullptr), inputs(), outputs() {}

    PtrNode(const ::tensorflow::NodeDef& src_node, const std::map<std::string, PtrNode*>& node_dictionary) {
        node = &src_node;
        std::vector<std::string> parsedName;
        for (const auto& input_name : node->input()) {
            parse_node_name(input_name, parsedName);

            auto input_node = node_dictionary.find(parsedName[0]);
            if (input_node == node_dictionary.end()) {
                continue;
            }

            input_node->second->outputs.push_back(this);
            inputs.push_back(input_node->second);
        }
    }

    void find_parent_by_op(const std::string& op, std::vector<PtrNode*>& result) const {
        for (auto input : inputs) {
            if (input->op() == op) {
                result.push_back(input);
            }
            input->find_parent_by_op(op, result);
        }
    }

    static void parse_node_name(const std::string& name, std::vector<std::string>& result) {
        result.clear();
        size_t left_pos = name.find_first_of('^'), right_pos = name.find(':');
        if (left_pos != std::string::npos && left_pos < right_pos) {
            ++left_pos;
        } else {
            left_pos = 0;
        }
        while (right_pos != std::string::npos && right_pos > left_pos) {
            result.push_back(name.substr(left_pos, right_pos - left_pos));
            left_pos = right_pos + 1;
            right_pos = name.find(':', left_pos);
        }
        result.push_back(name.substr(left_pos, name.length() - left_pos));
    }

    const std::string& op() const {
        return node->op();
    }
};

static void read_stateful_partitioned_call(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                                           const ::tensorflow::NodeDef& partCall,
                                           std::map<std::string, PtrNode*>& node_dictionary) {
    FRONT_END_GENERAL_CHECK(partCall.op() == "StatefulPartitionedCall", "Passed node isn't StatefulPartitionedCall");

    std::string func_name = partCall.attr().at("f").func().name();

    const ::tensorflow::FunctionDef* func_def = nullptr;
    for (const auto& func : graph_def->library().function()) {
        if (func.signature().name() == func_name) {
            func_def = &func;
            break;
        }
    }

    FRONT_END_GENERAL_CHECK(func_def, "Function isn't found in the library");
    FRONT_END_GENERAL_CHECK(graph_def->has_library(), "GraphDef contains functions, but doesn't have the library");

    std::map<std::string, PtrNode*> nodes;

    // Filling temporary input nodes for exact function
    for (int i = 0; i < func_def->signature().input_arg_size(); ++i) {
        const auto& input_arg = func_def->signature().input_arg(i).name();
        const auto& parent_input = partCall.input(i);
        auto input_node = node_dictionary.find(parent_input);
        if (input_node != node_dictionary.end()) {
            nodes[input_arg] = input_node->second;
        }
    }

    // Parsing nodes and inline partitioned calls
    for (const auto& node : func_def->node_def()) {
        nodes[node.name()] = new PtrNode(node, nodes);

        if (node.op() == "StatefulPartitionedCall") {
            read_stateful_partitioned_call(graph_def, node, nodes);
        }
    }

    // Removing temporary input nodes
    for (int i = 0; i < func_def->signature().input_arg_size(); ++i) {
        const auto& input_arg = func_def->signature().input_arg(i).name();
        auto input_node = nodes.find(input_arg);
        if (input_node != nodes.end()) {
            nodes.erase(input_node);
        }
    }

    // Moving nodes to the global dictionary
    for (const auto& node : nodes) {
        std::string global_name = partCall.name() + "/" + node.first;
        node_dictionary[global_name] = node.second;
    }
}

void GraphIteratorSavedModel::map_assignvariable(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                                                 std::map<std::string, std::string>& variables_map) const {
    std::map<std::string, PtrNode*> nodes;

    for (const auto& node : graph_def->node()) {
        nodes[node.name()] = new PtrNode(node, nodes);

        if (node.op() == "StatefulPartitionedCall") {
            read_stateful_partitioned_call(graph_def, node, nodes);
        }
    }

    for (const auto& node : nodes) {
        if (node.second->op() != "AssignVariableOp") {
            continue;
        }

        // TODO: assets reading

        std::vector<PtrNode*> restorev2_nodes;
        std::vector<PtrNode*> varhandle_nodes;

        node.second->find_parent_by_op("RestoreV2", restorev2_nodes);
        node.second->find_parent_by_op("VarHandleOp", varhandle_nodes);

        FRONT_END_GENERAL_CHECK(restorev2_nodes.size() == 1, "Found unexpected amount of RestoreV2 nodes");
        FRONT_END_GENERAL_CHECK(varhandle_nodes.size() == 1, "Found unexpected amount of VarHandleOp nodes");

        std::vector<std::string> restore_output;
        // Expected path is: RestoreV2 -(output_index)-(0)-> Identity -(0)-(1)-> AssignVariableOp
        PtrNode::parse_node_name(node.second->inputs[1]->node->input(0), restore_output);

        int output_index = std::atoi(restore_output[restore_output.size() - 1].c_str());

        // Expected path is: Const(tensor_names) -(0)-(1)-> RestoreV2
        const auto& variable_name =
            restorev2_nodes[0]->inputs[1]->node->attr().at("value").tensor().string_val(output_index);

        variables_map[varhandle_nodes[0]->node->name()] = variable_name;
    }

    nodes.clear();
}

bool GraphIteratorSavedModel::is_supported(const std::string& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join({path, "saved_model.pb"}));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool GraphIteratorSavedModel::is_supported(const std::wstring& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join_w({path, L"saved_model.pb"}));
}
#endif

template <>
std::basic_string<char> get_saved_model_name<char>() {
    return "/saved_model.pb";
}
template <>
std::basic_string<char> get_variables_index_name<char>() {
    return "/variables/variables.index";
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_saved_model_name<wchar_t>() {
    return L"/saved_model.pb";
}
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>() {
    return L"/variables/variables.index";
}
#endif

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
