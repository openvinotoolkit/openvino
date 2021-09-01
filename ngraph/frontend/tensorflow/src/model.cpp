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

    initial_traverse_graph();
}

std::vector<Place::Ptr> InputModelTensorflow::get_inputs() const {
    return m_inputs;
}

std::vector<Place::Ptr> InputModelTensorflow::get_outputs() const {
    return m_outputs;
}

void InputModelTensorflow::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& pshape) {
    auto place_tf = std::dynamic_pointer_cast<PlaceTF>(place);
    partialShapes[place_tf->get_names()[0]] = pshape;
}

ngraph::PartialShape InputModelTensorflow::get_partial_shape(Place::Ptr place) const {
    auto place_tf = std::dynamic_pointer_cast<PlaceTF>(place);
    ngraph::PartialShape result_shape;
    // TODO: replace by node cache without going through all nodes each time
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto node = graph_impl->get();
        if (node->name() == place_tf->get_names()[0]) {
            node->getAttrValue2("shape", &result_shape);
            break;
        }
    }
    // WARNING! Redesign GraphIterator -- it is not really good thing, detach an iterator from graph itself
    graph_impl->reset();
    return result_shape;
}

std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> InputModelTensorflow::get_ops() const {
    // TODO: call that ONLY if model modified
    return determine_cut_nodes();
}

void InputModelTensorflow::initial_traverse_graph() {
    std::set<std::string> all_names;
    std::set<std::string> names_with_consumers;

    m_inputs.clear();
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto op = graph_impl->get();
        all_names.insert(op->name());
        m_ops_topology_sorted.push_back(std::make_shared<OpPlaceTF>(*this, op));
        m_ops[op->name()] = m_ops_topology_sorted.back();
        if (graph_impl->get()->op() == "Placeholder") {
            m_inputs.push_back(m_ops_topology_sorted.back());
        }
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

std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> InputModelTensorflow::determine_cut_nodes() const {
    std::queue<tensorflow::detail::TFNodeDecoder*> q;
    std::unordered_set<std::string> visited;
    std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> new_ops;
    for (const auto& output_op : m_outputs) {
        auto op_name = output_op->get_names()[0];
        if (!visited.count(op_name)) {
            visited.insert(op_name);
            auto out_op_place = std::dynamic_pointer_cast<ngraph::frontend::OpPlaceTF>(output_op);
            if (out_op_place) {
                // TODO: throw if nullptr
                new_ops.push_back(out_op_place);
                q.push(out_op_place->get_desc().get());
            }
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
                q.push(op_it->second->get_desc().get());
            }
        }
    }
    return new_ops;
}
