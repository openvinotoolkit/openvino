//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/computation_builder.hpp"
#include "simulation/layers_reader.hpp"
#include "simulation/operations.hpp"
#include "simulation/performance_mode.hpp"
#include "simulation/simulation.hpp"

#include "utils/error.hpp"

#include <opencv2/gapi/streaming/meta.hpp>

struct OpBuilder {
    void build(NodeHandle nh, const Infer& infer);
    void build(NodeHandle nh, const Delay& delay);
    void build(NodeHandle nh, const Compound& compound);

    Graph& graph;
    IBuildStrategy::Ptr strategy;
    const InferenceParamsMap& params_map;
};

void OpBuilder::build(NodeHandle nh, const Compound& compound) {
    // Retrieving destination nodes of the current node nh
    auto out_nhs = nh->dstNodes();

    // NB: The Dummy node ensures proper handling of multiple inputs
    auto dummy_nh = graph.create();
    auto provider = std::make_shared<CircleBuffer>(utils::createRandom({1}, CV_8U));
    DummyCall dummy_call{{provider}, 0};
    graph.meta(dummy_nh).set(GOperation{std::move(dummy_call)});
    auto in_nhs = nh->srcNodes();

    // removing input edges to go through dummy node and not to compound node
    auto src_edges = nh->srcEdges();
    for (size_t i = 0; i < src_edges.size(); ++i) {
        graph.remove(src_edges[i]);
    }

    for (uint32_t i = 0; i < in_nhs.size(); ++i) {
        graph.meta(graph.link(in_nhs[i], dummy_nh)).set(InputIdx{i});  // Linking in_nhs with dummy_nh
    }

    auto dummy_out_nh = graph.create();  // Creating output dunmmy node
    graph.meta(graph.link(dummy_nh, dummy_out_nh))
            .set(OutputIdx{0u});  // linking dummy node handle and output dummy node handle
    graph.meta(dummy_out_nh).set(GData{});
    graph.meta(graph.link(dummy_out_nh, nh)).set(InputIdx{0u});

    ASSERT(nh->dstEdges().size() == 1u);
    auto dst_edge = nh->dstEdges().front();
    graph.meta(dst_edge).set(OutputIdx{0u});

    graph.meta(graph.link(nh, out_nhs.front())).set(OutputIdx{0u});

    ModelsAttrMap<std::string> input_data_map;
    ModelsAttrMap<IRandomGenerator::Ptr> initializers_map;

    for (const auto& [tag, params] : compound.infer_params) {
        input_data_map[tag];
        initializers_map[tag];
    }

    PerformanceSimulation::Options opts{
            nullptr,  // global_initializer
            initializers_map,
            input_data_map,
            true,  // inference_only
            {}     // target latency
    };

    Simulation::Config cfg{compound.tag,
                           0u,     // frames_interval_in_ms
                           false,  // disable_high_resolution_timer
                           compound.subgraph, compound.infer_params};

    auto compiled = std::make_shared<PerformanceSimulation>(std::move(cfg), std::move(opts))
                            ->compileSync(false /*drop_frames*/);
    auto term_criterion = std::make_shared<Iterations>(compound.repeat_count);
    auto f = [compiled, term_criterion]() {
        compiled->run(term_criterion);
    };

    CompoundCall compound_call{f};
    graph.meta(nh).set(GOperation{std::move(compound_call)});
}

void OpBuilder::build(NodeHandle nh, const Delay& delay) {
    auto in_nhs = nh->srcNodes();
    auto out_nhs = nh->dstNodes();
    // FIXME: Once nh is removed, delay info is no longer alive!!!
    const auto time_in_us = delay.time_in_us;
    graph.remove(nh);

    auto delay_nh = graph.create();
    auto provider = std::make_shared<CircleBuffer>(utils::createRandom({1}, CV_8U));
    graph.meta(delay_nh).set(GOperation{DummyCall{{provider}, time_in_us}});

    for (uint32_t i = 0; i < in_nhs.size(); ++i) {
        graph.meta(graph.link(in_nhs[i], delay_nh)).set(InputIdx{i});
    }
    graph.meta(graph.link(delay_nh, out_nhs.front())).set(OutputIdx{0u});
}

void OpBuilder::build(NodeHandle nh, const Infer& infer) {
    const auto& params = params_map.at(infer.tag);
    auto [in_layers, out_layers] = LayersReader::readLayers(params);
    InferDesc desc{infer.tag, std::move(in_layers), std::move(out_layers)};

    auto out_nhs = nh->dstNodes();
    ASSERT(out_nhs.size() == 1);

    auto [providers, in_meta, out_meta, disable_copy] = strategy->build(desc);
    ASSERT(providers.size() == desc.input_layers.size());
    ASSERT(in_meta.size() == desc.input_layers.size());
    ASSERT(out_meta.size() == desc.output_layers.size());

    // NB: Check if some of the Delay's was fused to this Infer
    uint64_t delay_in_us = 0u;
    if (graph.meta(nh).has<Delay>()) {
        delay_in_us = graph.meta(nh).get<Delay>().time_in_us;
    }

    auto dummy_nh = graph.create();
    DummyCall dummy_call{providers, delay_in_us, disable_copy};
    graph.meta(dummy_nh).set(GOperation{std::move(dummy_call)});
    auto in_nhs = nh->srcNodes();
    for (uint32_t i = 0; i < in_nhs.size(); ++i) {
        graph.meta(graph.link(in_nhs[i], dummy_nh)).set(InputIdx{i});
    }

    graph.remove(nh);

    auto infer_nh = graph.create();
    for (uint32_t layer_idx = 0; layer_idx < desc.input_layers.size(); ++layer_idx) {
        // NB: Create dummy out node and link with dummy.
        auto dummy_out_nh = graph.create();
        graph.meta(dummy_out_nh) += std::move(in_meta[layer_idx]);
        graph.meta(graph.link(dummy_nh, dummy_out_nh)).set(OutputIdx{layer_idx});
        graph.meta(dummy_out_nh).set(GData{});
        // NB: Finally link dummy out with infer
        graph.meta(graph.link(dummy_out_nh, infer_nh)).set(InputIdx{layer_idx});
    }

    auto out_nh = out_nhs.front();
    graph.meta(graph.link(infer_nh, out_nh)).set(OutputIdx{0u});
    graph.meta(out_nh) += out_meta.front();
    for (uint32_t layer_idx = 1; layer_idx < desc.output_layers.size(); ++layer_idx) {
        auto infer_out_nh = graph.create();
        graph.meta(infer_out_nh) = std::move(out_meta[layer_idx]);
        graph.meta(infer_out_nh).set(GData{});
        graph.meta(graph.link(infer_nh, infer_out_nh)).set(OutputIdx{layer_idx});
    }

    InferCall infer_call{desc.tag, extractLayerNames(desc.input_layers), extractLayerNames(desc.output_layers)};
    graph.meta(infer_nh).set(GOperation{std::move(infer_call)});
};

static bool fuseDelay(Graph& graph, NodeHandle nh, const Delay& delay) {
    // NB: Current fusing is trivial and applied only for the following case:
    // 1) Delay has only single Infer reader
    // 2) Infer doesn't have any other writers except Delay
    // e.g: [Delay] -> (out) -> [Infer]

    // NB: Access readers of delay output data node.
    auto delay_out_nh = nh->dstNodes().front();
    auto out_edges = delay_out_nh->dstEdges();
    // NB: Don't fuse Delay either if it has multiple readers
    // or doesn't have readers at all (1)
    if (out_edges.size() != 1u) {
        return false;
    }

    auto out_edge = out_edges.front();
    auto op_nh = out_edge->dstNode();
    auto op = graph.meta(op_nh).get<Op>().kind;
    // NB: Don't fuse Delay if reader either not an Infer (1)
    // or it has other writers except Delay (2).
    if (!std::holds_alternative<Infer>(op) || op_nh->srcEdges().size() != 1u) {
        // TODO: Can be also fused to another "delay".
        return false;
    }

    // NB: Fuse the Delay into Infer:
    // 1) Assign Delay meta directly to Infer
    // 2) Remove Delay node
    // 3) Redirect Delay writers to Infer
    graph.meta(op_nh).set(delay);
    for (auto in_nh : nh->srcNodes()) {
        graph.link(in_nh, op_nh);
    }
    graph.remove(nh);
    graph.remove(delay_out_nh);

    return true;
}

struct Protocol {
    cv::GProtoArgs graph_inputs;
    cv::GProtoArgs graph_outputs;
};

enum class NodeState { EXPLORING, VISITED };

static void visit(NodeHandle nh, std::unordered_map<NodeHandle, NodeState>& state) {
    auto curr_node_it = state.emplace(nh, NodeState::EXPLORING).first;
    for (const auto& dst_nh : nh->dstNodes()) {
        const auto dst_it = state.find(dst_nh);
        if (dst_it == state.end()) {
            visit(dst_nh, state);
        } else if (dst_it->second == NodeState::EXPLORING) {
            THROW_ERROR("Scenario graph has a cycle!");
        }
    }
    curr_node_it->second = NodeState::VISITED;
};

namespace passes {

// NB: Throw an exception if there is a cycle in graph
void throwIfCycle(Graph& graph) {
    std::unordered_map<NodeHandle, NodeState> state;
    for (const auto& nh : graph.nodes()) {
        if (state.find(nh) == state.end()) {
            visit(nh, state);
        }
    }
}

// NB: Determines what would be the computation graph
// inputs and outputs and marks intermediate data nodes
void init(Graph& graph) {
    ASSERT(!graph.nodes().empty());
    uint32_t num_sources = 0;
    for (auto nh : graph.nodes()) {
        if (graph.meta(nh).has<Source>()) {
            ++num_sources;
            graph.meta(nh).set(GraphInput{});
        } else {
            // NB: Check that graph is connected
            ASSERT(!nh->srcNodes().empty());
        }
        if (nh->dstNodes().empty()) {
            ASSERT(graph.meta(nh).has<Data>());
            graph.meta(nh).set(GraphOutput{});
        }
        if (!graph.meta(nh).has<Op>()) {
            ASSERT(graph.meta(nh).has<Data>());
            graph.meta(nh).set(GData{});
        }
    }
    ASSERT(num_sources != 0);
};

// NB: Fuses delay to the inference nodes as the delay can be performed
// as part of the model dummy preprocessing
void fuseDelays(Graph& graph) {
    // NB: Iterate over graph nodes until all delays are fused.
    while (true) {
        bool is_fused = false;
        for (auto nh : graph.nodes()) {
            if (!graph.meta(nh).has<Op>()) {
                continue;
            }
            auto op = graph.meta(nh).get<Op>().kind;
            if (std::holds_alternative<Delay>(op)) {
                auto delay = std::get<Delay>(op);
                if (fuseDelay(graph, nh, delay)) {
                    is_fused = true;
                    break;
                }
            }
        }
        // NB: If delay was fused, some of the nodes were removed
        // Iterate one more time...
        if (!is_fused) {
            break;
        }
    }
};

// NB: Finds the maximum parallelism depth to tell concurrent executor
// how many threads should be used for execution
void findMaxParallelBranches(Graph& graph, uint32_t& max_parallel_branches) {
    // NB: Basically the maximum parallelism in computational graph
    // is the maximum width of its level in BFS traversal, taking into
    // account that dependencies for the node are resolved
    std::unordered_set<NodeHandle> curr_lvl;
    for (auto nh : graph.nodes()) {
        if (graph.meta(nh).has<Source>()) {
            for (auto op_nh : nh->dstNodes()) {
                curr_lvl.emplace(op_nh);
            }
        }
    }

    std::unordered_set<NodeHandle> visited;

    auto get_all_deps = [&](auto nh) {
        std::unordered_set<NodeHandle> deps;
        for (auto in_nhs : nh->srcNodes()) {
            for (auto op_nhs : in_nhs->srcNodes()) {
                deps.emplace(op_nhs);
            }
        }
        return deps;
    };

    auto all_deps_resolved = [&](auto nh) {
        auto deps = get_all_deps(nh);
        return std::all_of(deps.begin(), deps.end(), [&](auto dep) {
            return visited.find(dep) != visited.end();
        });
    };

    max_parallel_branches = static_cast<uint32_t>(curr_lvl.size());
    while (!curr_lvl.empty()) {
        std::unordered_set<NodeHandle> next_lvl;
        for (auto nh : curr_lvl) {
            visited.emplace(nh);
            ASSERT(nh->dstNodes().size() == 1u);
            auto data_nh = nh->dstNodes().front();
            for (auto op_nh : data_nh->dstNodes()) {
                if (all_deps_resolved(op_nh)) {
                    next_lvl.emplace(op_nh);
                }
            }
        }
        if (next_lvl.size() > max_parallel_branches) {
            max_parallel_branches = static_cast<uint32_t>(next_lvl.size());
        }
        curr_lvl = std::move(next_lvl);
    }
}

// NB: Build "G" operations according to scenario graph nodes
void buildOperations(Graph& graph, IBuildStrategy::Ptr strategy, const InferenceParamsMap& params_map) {
    OpBuilder builder{graph, strategy, params_map};
    for (auto nh : graph.nodes()) {
        // NB: Skip data nodes
        if (!graph.meta(nh).has<Op>()) {
            continue;
        }
        std::visit(
                [nh, &builder](const auto& op) {
                    builder.build(nh, op);
                },
                graph.meta(nh).get<Op>().kind);
    }

    for (auto nh : graph.nodes()) {
        // NB: Make sure all data nodes that needs to be
        // dumped or validated are graph outputs.
        if (!graph.meta(nh).has<GraphOutput>() && (graph.meta(nh).has<Validate>() || graph.meta(nh).has<Dump>())) {
            graph.meta(nh).set(GraphOutput{});
        }
    }
};

void buildComputation(Graph& graph, Protocol& proto) {
    cv::GProtoArgs graph_inputs;
    cv::GProtoArgs graph_outputs;

    std::unordered_map<NodeHandle, cv::GProtoArg> all_data;
    auto sorted = graph.sorted();

    // NB: Initialize "G" inputs
    for (auto nh : sorted) {
        if (graph.meta(nh).has<GraphInput>()) {
            auto it = all_data.emplace(nh, cv::GProtoArg{cv::GMat()}).first;
            graph_inputs.push_back(it->second);
        }
    }
    // NB: Apply "G" operations in topological order
    for (auto nh : sorted) {
        if (graph.meta(nh).has<GOperation>()) {
            const auto& operation = graph.meta(nh).get<GOperation>();
            // NB: Map input args to the correct input index.
            std::unordered_map<uint32_t, cv::GProtoArg> idx_to_arg;
            auto in_ehs = nh->srcEdges();
            for (auto in_eh : in_ehs) {
                ASSERT(graph.meta(in_eh).has<InputIdx>());
                const uint32_t in_idx = graph.meta(in_eh).get<InputIdx>().idx;
                auto arg = all_data.at(in_eh->srcNode());
                idx_to_arg.emplace(in_idx, arg);
            }
            cv::GProtoArgs in_args;
            for (uint32_t idx = 0; idx < idx_to_arg.size(); ++idx) {
                in_args.push_back(idx_to_arg.at(idx));
            }
            // NB: Link G-API operation with its io data.
            auto out_args = operation.on(in_args);
            // TODO: Validation in/out amount and types...
            // NB: Map output args to the correct index.
            auto out_ehs = nh->dstEdges();
            for (auto out_eh : out_ehs) {
                ASSERT(graph.meta(out_eh).has<OutputIdx>());
                const uint32_t out_idx = graph.meta(out_eh).get<OutputIdx>().idx;
                auto out_nh = out_eh->dstNode();
                all_data.emplace(out_nh, out_args[out_idx]);
            }
        }
    }

    // NB: Collect "G" outputs
    for (auto nh : graph.nodes()) {
        if (graph.meta(nh).has<GraphOutput>()) {
            graph_outputs.push_back(all_data.at(nh));
        }
    }

    ASSERT(!graph_inputs.empty())
    ASSERT(!graph_outputs.empty())
    // NB: Finally save computation i/o to build GComputation later on
    proto = Protocol{std::move(graph_inputs), std::move(graph_outputs)};
}

static void collectOutputMeta(Graph& graph, std::vector<Meta>& out_meta) {
    for (auto nh : graph.nodes()) {
        if (graph.meta(nh).has<GraphOutput>()) {
            out_meta.push_back(graph.meta(nh));
        }
    }
}

}  // namespace passes

ComputationBuilder::ComputationBuilder(IBuildStrategy::Ptr strategy): m_strategy(strategy) {
}

Computation ComputationBuilder::build(ScenarioGraph& graph, const InferenceParamsMap& infer_params,
                                      const ComputationBuilder::Options& opts) {
    uint32_t max_parallel_branches = 1u;
    auto compile_args = cv::compile_args(cv::gapi::kernels<GCPUDummyM, GCPUCompound>());
    std::vector<Meta> outputs_meta;
    Protocol proto;

    using namespace std::placeholders;
    graph.pass(passes::throwIfCycle);
    graph.pass(passes::init);
    graph.pass(passes::fuseDelays);
    graph.pass(std::bind(passes::findMaxParallelBranches, _1, std::ref(max_parallel_branches)));
    graph.pass(std::bind(passes::buildOperations, _1, m_strategy, std::cref(infer_params)));
    graph.pass(std::bind(passes::buildComputation, _1, std::ref(proto)));
    graph.pass(std::bind(passes::collectOutputMeta, _1, std::ref(outputs_meta)));

    if (opts.add_perf_meta) {
        // FIXME: Must work with any G-Type!
        ASSERT(cv::util::holds_alternative<cv::GMat>(proto.graph_outputs.front()));
        cv::GMat g = cv::util::get<cv::GMat>(proto.graph_outputs.front());
        proto.graph_outputs.emplace_back(cv::gapi::streaming::timestamp(g).strip());
        proto.graph_outputs.emplace_back(cv::gapi::streaming::seq_id(g).strip());
    }

    cv::GComputation comp(cv::GProtoInputArgs{std::move(proto.graph_inputs)},
                          cv::GProtoOutputArgs{std::move(proto.graph_outputs)});

    return Computation{std::move(comp), std::move(compile_args), std::move(outputs_meta), {max_parallel_branches}};
}
