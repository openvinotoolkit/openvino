// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <numeric>
#include <queue>
#include <tensorflow_frontend/model.hpp>
#include <tensorflow_frontend/place.hpp>

//#include "graph.pb.h"
//#include "tensor.pb.h"

#include <ngraph/pass/manager.hpp>

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"
//#include <ngraph/pass/transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>

#include "default_opset.h"
#include "graph.hpp"
#include "ngraph_builder.h"
#include "ngraph_conversions.h"

using namespace google;

using namespace ngraph::frontend;

using ::tensorflow::GraphDef;
using ::tensorflow::ngraph_bridge::GraphIteratorProto;

InputModelTensorflow::InputModelTensorflow(const std::string& _path) : path(_path) {
    std::ifstream pb_stream(path, std::ios::binary);
    graph_def = std::make_shared<GraphDef>();
    std::cout << "[ INFO ] Model Parsed: " << graph_def->ParseFromIstream(&pb_stream) << std::endl;
    std::cout << "[ INFO ] Loaded model contains " << graph_def->node_size() << " nodes." << std::endl;
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(graph_def.get());

    determine_outputs();
}

std::vector<Place::Ptr> InputModelTensorflow::get_inputs() const {
    std::vector<Place::Ptr> result;
    for (; !graph_impl->is_end(); graph_impl->next()) {
        std::cout << "graph_impl->get()->op() = " << graph_impl->get()->op() << "\n";
        if (graph_impl->get()->op() == "Placeholder")
            result.push_back(std::make_shared<PlaceTensorflow>(graph_impl->get()->name()));
    }
    graph_impl->reset();
    return result;
}

void InputModelTensorflow::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& pshape) {
    auto place_tf = std::dynamic_pointer_cast<PlaceTensorflow>(place);
    partialShapes[place_tf->name] = pshape;
}

ngraph::PartialShape InputModelTensorflow::get_partial_shape(Place::Ptr place) const {
    auto place_tf = std::dynamic_pointer_cast<PlaceTensorflow>(place);
    ngraph::PartialShape result_shape;
    // TODO: replace by node cache without going through all nodes each time
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto node = graph_impl->get();
        if (node->name() == place_tf->name) {
            node->getAttrValue2("shape", &result_shape);
            break;
        }
    }
    // WARNING! Redesign GraphIterator -- it is not really good thing, detach an iterator from graph itself
    graph_impl->reset();
    return result_shape;
}


std::vector<std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder>> InputModelTensorflow::get_ops() {
    // TODO: call that ONLY if model modified
    return determine_cut_nodes();
}


void InputModelTensorflow::determine_outputs() {
    std::set<std::string> all_names;
    std::set<std::string> names_with_consumers;
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto op = graph_impl->get();
        all_names.insert(op->name());
        m_ops[op->name()] = op;
        m_ops_topology_sorted.push_back(op);
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            std::string input_name;
            size_t port_idx;
            try {
                op->input_node(i, &input_name, &port_idx);
                names_with_consumers.insert(input_name);
            } catch (const std::exception& e) {
                std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                          << "', expected input name: '" << input_name << "', expected input port index: " << port_idx
                          << '\n';
                throw;
            }
        }
    }
    std::set<std::string> names_without_consumers;
    std::set_difference(all_names.begin(),
                        all_names.end(),
                        names_with_consumers.begin(),
                        names_with_consumers.end(),
                        std::inserter(names_without_consumers, names_without_consumers.begin()));
    graph_impl->reset();

    m_outputs.clear();
    for (auto& out_name : names_without_consumers) {
        m_outputs.push_back(m_ops[out_name]);
    }
}

std::vector<std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder>>
InputModelTensorflow::determine_cut_nodes() const {
    std::queue<tensorflow::detail::TFNodeDecoder*> q;
    std::unordered_set<std::string> visited;
    std::vector<std::shared_ptr<tensorflow::detail::TFNodeDecoder>> new_ops;
    for (const auto& output_op : m_outputs) {
        if (!visited.count(output_op->name())) {
            visited.insert(output_op->name());
            new_ops.push_back(output_op);
            q.push(output_op.get());
        }
    }
    while (!q.empty()) {
        auto op = q.front();
        q.pop();
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            std::string input_name;
            size_t port_idx;
            try {
                op->input_node(i, &input_name, &port_idx);
            } catch (const std::exception& e) {
                std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                          << "', expected input name: '" << input_name << "', expected input port index: " << port_idx
                          << '\n';
                throw;
            }
            auto op_it = m_ops.find(input_name);
            if (op_it != m_ops.end() && !visited.count(input_name)) {
                visited.insert(input_name);
                new_ops.push_back(op_it->second);
                // TODO: check that op is not input or frozen
                q.push(op_it->second.get());
            }
        }
    }
    return new_ops;
}
