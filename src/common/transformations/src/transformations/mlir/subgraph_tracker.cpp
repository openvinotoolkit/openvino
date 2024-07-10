// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <unordered_map>

#include "subgraph_tracker.hpp"


namespace ov {
namespace mlir {


void Subgraph::merge (Subgraph& other) {
    nodes.insert(nodes.end(), other.nodes.begin(), other.nodes.end());
}


SubgraphTracker::SubgraphTracker(Finalizer finalizer): m_finalizer(finalizer) {}

void SubgraphTracker::add_node (NodePtr node, bool belongs) {
    // collect all subgraph ids that input nodes belong to and all dependencies
    Dependencies input_subgraphs;
    Dependencies input_dependencies;
    for(auto input_value: node->input_values()) {
        auto node = input_value.get_node_shared_ptr();
        if(auto id = get_subgraph_id(node)) {
            input_subgraphs.insert(ov::symbol::ancestor_of(id));
        }
        const auto& deps = get_dependencies(node);
        for(auto dep: deps) {
            input_dependencies.insert(ov::symbol::ancestor_of(dep));
        }
    }

    if(belongs) {
        // Below we refuse to merge subgraphs if all of them cannot merge to a single subgraph, this is rough because
        // there are cases when part of the input subgraphs can consume the node and others will come as inputs -- TODO.
        // TODO: leave only those input subgraphs that are not conflicting with other subgraphs nor with any dependencies
        if(input_subgraphs.empty() || intersected(input_subgraphs, input_dependencies)) {   // no input subgraphs || cannot merge all due to cycles
            try_terminate_subgraphs(input_subgraphs, node);

            // start a new subgraph
            auto subgraph_id = new_subgraph();
            add_node_to_subgraph(node, subgraph_id);
            set_subgraph_id(node, subgraph_id);
            input_dependencies.insert(input_subgraphs.begin(), input_subgraphs.end());
        } else {
            auto merged_subgraph_id = std::accumulate(
                input_subgraphs.begin(),
                input_subgraphs.end(),
                *input_subgraphs.begin(),
                [this](SubgraphID a, SubgraphID b) {
                    merge_subgraphs(a, b);
                    return a;
                }
            );
            set_subgraph_id(node, merged_subgraph_id);
            add_node_to_subgraph(node, merged_subgraph_id);
        }

    } else {
        try_terminate_subgraphs(input_subgraphs, node);
        set_subgraph_id(node, nullptr);
        input_dependencies.insert(input_subgraphs.begin(), input_subgraphs.end());
    }
    set_dependencies(node, input_dependencies);
}

void SubgraphTracker::finalize() {
    for(auto subgraph_record: m_subgraphs) {
        terminate_subgraph(subgraph_record.first);
    }
}

SubgraphID SubgraphTracker::new_subgraph() {
    SubgraphID id = std::make_shared<ov::Symbol>();
    m_subgraphs[id] = std::make_shared<Subgraph>();
    return id;
}

void SubgraphTracker::add_node_to_subgraph(NodePtr node, SubgraphID id) {
    get_subgraph(id)->nodes.push_back(node);
}

void SubgraphTracker::merge_subgraphs(SubgraphID id1, SubgraphID id2) {
    id1 = ov::symbol::ancestor_of(id1);
    id2 = ov::symbol::ancestor_of(id2);
    if (id1 == id2) return;

    auto subgraph1 = get_subgraph(id1);
    auto subgraph2 = get_subgraph(id2);
    subgraph1->merge(*subgraph2);
    m_subgraphs.erase(id1);
    m_subgraphs.erase(id2);
    ov::symbol::set_equal(id1, id2);
    id1 = ov::symbol::ancestor_of(id1);
    m_subgraphs[id1] = subgraph1;
}

SubgraphPtr SubgraphTracker::get_subgraph(SubgraphID id) {
    return m_subgraphs.at(ov::symbol::ancestor_of(id));
}

// set/get all subgraph ids that contribute to a given node

const SubgraphTracker::Dependencies& SubgraphTracker::get_dependencies(NodePtr node) {
    return node->get_rt_info().at("__subgraph_dependencies").as<Dependencies>();
}
void SubgraphTracker::set_dependencies(NodePtr node, const Dependencies& dependencies) {
    node->get_rt_info()["__subgraph_dependencies"] = dependencies;
}

// set/get subgraph id that a give node belongs to

SubgraphID SubgraphTracker::get_subgraph_id(NodePtr node) {
    auto id = node->get_rt_info().at("__subgraph_id").as<SubgraphID>();
    if(id) {
        id = ov::symbol::ancestor_of(id);
    }
    return id;
}

void SubgraphTracker::set_subgraph_id(NodePtr node, SubgraphID id) {
    node->get_rt_info()["__subgraph_id"] = id;
}

bool SubgraphTracker::intersected(const Dependencies& a, const Dependencies& b) {
    for(const auto& x: a) {
        if(b.count(x))
            return true;
    }
    return false;
}

void SubgraphTracker::terminate_subgraph(SubgraphID id) {
    id = ov::symbol::ancestor_of(id);
    auto subgraph = get_subgraph(id);
    // Build subgraph inputs and outputs
    std::set<ov::Output<ov::Node>> inputs;
    auto& outputs = subgraph->outputs;
    auto& output_consumers = subgraph->output_consumers;
    for(auto node: subgraph->nodes) {
        for(auto input: node->input_values()) {
            auto input_id = get_subgraph_id(input.get_node_shared_ptr());
            if(!ov::symbol::are_equal(id, input_id)) {
                inputs.insert(input);
            }
        }
        for(auto output: node->outputs()) {
            const auto& consumers = output.get_target_inputs();
            InputVector external_consumers;
            for(auto consumer: consumers) {
                auto consumer_id = get_subgraph_id(consumer.get_node()->shared_from_this());
                if(!ov::symbol::are_equal(id, consumer_id)) {
                    external_consumers.push_back(consumer);
                }
            }
            bool used_outside = !external_consumers.empty();
            if(used_outside) {
                outputs.push_back(output);
                output_consumers.push_back(external_consumers);
            }
        }
    }
    subgraph->inputs.assign(inputs.begin(), inputs.end());
    m_finalizer(subgraph);
}

void SubgraphTracker::try_terminate_subgraphs(const Dependencies& subgraphs, NodePtr terminator) {
    // TODO: Terminate subgraphs earlier when all terminating nodes are known
    // TODO: try to merge subgraphs if they are being terminated simultaniously
}


} // namespace mlir
} // namespace ov
