// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "decoder_argdef.hpp"
#include "decoder_proto.hpp"
#include "graph.pb.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class GraphIteratorProto : public GraphIterator {
    std::shared_ptr<::tensorflow::GraphDef> m_graph_def;
    std::shared_ptr<::tensorflow::FunctionDef> m_func_def;

    size_t node_index = 0;
    std::vector<std::shared_ptr<DecoderBase>> m_decoders;
    std::unordered_map<std::string, int> m_library_map;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;

public:
    GraphIteratorProto(const std::shared_ptr<::tensorflow::GraphDef>& graph_def,
                       const std::shared_ptr<::tensorflow::FunctionDef>& func_def,
                       const std::unordered_map<std::string, int>& library_map)
        : m_graph_def(graph_def),
          m_func_def(func_def),
          m_library_map(library_map) {
        auto nodes_size = m_func_def->node_def_size();
        auto input_size = m_func_def->signature().input_arg_size();
        auto output_size = m_func_def->signature().output_arg_size();
        auto ret_map = m_func_def->ret();

        // fill all inputs from library functions
        // these input_arg objects are of different type OpDef_ArgDef
        // they are not NodeDef so we use separate Decoder class
        for (int input_ind = 0; input_ind < input_size; ++input_ind) {
            auto input_arg = &m_func_def->signature().input_arg(input_ind);
            m_input_names.push_back(input_arg->name());
            m_decoders.push_back(std::make_shared<DecoderArgDef>(input_arg, "input_arg"));
        }

        // fill all node defs from library functions
        for (int node_ind = 0; node_ind < nodes_size; ++node_ind) {
            m_decoders.push_back(std::make_shared<DecoderProto>(&(m_func_def->node_def(node_ind))));
        }

        // fill all outputs from library functions
        // these output_arg objects are of different type OpDef_ArgDef
        // they are not NodeDef so we use separate Decoder class
        for (int output_ind = 0; output_ind < output_size; ++output_ind) {
            auto output_arg = &m_func_def->signature().output_arg(output_ind);
            m_output_names.push_back(output_arg->name());
            auto producer_name = ret_map.at(output_arg->name());
            m_decoders.push_back(std::make_shared<DecoderArgDef>(output_arg, "output_arg", producer_name));
        }
    }

    template <typename T>
    GraphIteratorProto(const std::basic_string<T>& path)
        : m_graph_def(std::make_shared<::tensorflow::GraphDef>()),
          m_func_def(nullptr) {
        std::ifstream pb_stream(path, std::ios::in | std::ifstream::binary);

        FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file does not exist");
        FRONT_END_GENERAL_CHECK(m_graph_def->ParseFromIstream(&pb_stream), "Model cannot be parsed");

        auto nodes_size = m_graph_def->node_size();
        m_decoders.resize(static_cast<size_t>(nodes_size));
        for (int node_ind = 0; node_ind < nodes_size; ++node_ind) {
            m_decoders[node_ind] = std::make_shared<DecoderProto>(&m_graph_def->node(node_ind));
        }

        // initialize a library map
        auto num_funcs = m_graph_def->library().function_size();
        for (int func_ind = 0; func_ind < num_funcs; ++func_ind) {
            auto func = m_graph_def->library().function(func_ind);
            auto func_name = func.signature().name();
            m_library_map.insert(std::pair<std::string, int>(func_name, func_ind));
        }
    }

    /// Set iterator to the start position
    void reset() override {
        node_index = 0;
    }

    size_t size() const override {
        return m_decoders.size();
    }

    /// Moves to the next node in the graph
    void next() override {
        node_index++;
    }

    bool is_end() const override {
        return node_index >= m_decoders.size();
    }

    /// Return NodeContext for the current node that iterator points to
    std::shared_ptr<DecoderBase> get_decoder() const override {
        return m_decoders[node_index];
    }

    std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const override {
        if (m_library_map.count(func_name)) {
            auto func_ind = m_library_map.at(func_name);
            auto func_size = m_graph_def->library().function_size();
            FRONT_END_GENERAL_CHECK(
                0 <= func_ind && func_ind < func_size,
                "[TensorFlow Error] Internal Error: incorrect library map to cache function indices by names.");

            auto func = m_graph_def->library().function(func_ind);
            auto func_ptr = std::make_shared<::tensorflow::FunctionDef>(func);
            return std::make_shared<GraphIteratorProto>(m_graph_def, func_ptr, m_library_map);
        }

        return nullptr;
    }

    std::vector<std::string> get_input_names() const {
        return m_input_names;
    }

    std::vector<std::string> get_output_names() const {
        return m_output_names;
    }
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
