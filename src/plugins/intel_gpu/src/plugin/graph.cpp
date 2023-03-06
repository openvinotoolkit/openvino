// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/profiling.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/plugin/infer_request.hpp"
#include "intel_gpu/runtime/itt.hpp"

#include <description_buffer.hpp>
#include <threading/ie_executor_manager.hpp>
#include <exec_graph_info.hpp>

#include <ie_ngraph_utils.hpp>

#include <list>
#include <set>
#include <unordered_set>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <utility>
#include <sys/types.h>
#include <sys/stat.h>
#include <exec_graph_info.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

Graph::Graph(InferenceEngine::CNNNetwork& network, RemoteContextImpl::Ptr context, const ExecutionConfig& config, uint16_t stream_id)
    : m_context(context)
    , m_networkName(network.getName())
    , m_config(config)
    , m_stream_id(stream_id)
    , m_state(0) {
    m_program = std::make_shared<Program>(network, get_engine(), config);
    if (m_program->m_max_batch > 1)
        m_config.set_property(ov::intel_gpu::max_dynamic_batch(m_program->m_max_batch));
    Build();
}

Graph::Graph(cldnn::BinaryInputBuffer &ib, RemoteContextImpl::Ptr context, const ExecutionConfig& config, uint16_t stream_id)
    : m_context(context)
    , m_config(config)
    , m_stream_id(stream_id)
    , m_state(0) {
    m_program = std::make_shared<Program>(get_engine(), config);
    if (m_program->m_max_batch > 1)
        m_config.set_property(ov::intel_gpu::max_dynamic_batch(m_program->m_max_batch));

    bool need_onednn_engine = false;
    ib >> need_onednn_engine;
    if (need_onednn_engine) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        get_engine().create_onednn_engine(config);
#else
        IE_THROW() << "[GPU] Current model cache requires OneDNN, but cannot use it.";
#endif  // ENABLE_ONEDNN_FOR_GPU
    }

    ib >> m_program->inputLayouts;
    Program::variables_state_info_map variablesStateInfoMap;
    ib >> variablesStateInfoMap;
    for (const auto& variablesStateInfo : variablesStateInfoMap) {
        m_program->AddVariableStateInfo(variablesStateInfo.first, *variablesStateInfo.second.begin());
    }
    ib >> primitiveIDs;
    ib >> prevPrimitiveIDs;
    ib >> profilingIDs;
    {
        size_t perfMap_size;
        ib >> perfMap_size;
        for (size_t i = 0; i < perfMap_size; ++i) {
            cldnn::primitive_id prim_id;
            ib >> prim_id;
            perfMap[prim_id].first = prim_id;
            auto& perfEntry = perfMap[prim_id].second;
            ib >> perfEntry.layerType;
            ib >> cldnn::make_data(&perfEntry.status, sizeof(InferenceEngine::InferenceEngineProfileInfo::LayerStatus));
            perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
            ib >> perfEntry.isCPU;
            ib >> perfEntry.parentPrimitive;
        }
    }
    ib >> outputDims;

    size_t num_networks;
    ib >> num_networks;
    for (size_t i = 0; i < num_networks; ++i) {
        m_networks.emplace_back(std::make_shared<cldnn::network>(ib, get_engine().create_stream(config), get_engine(), m_stream_id));
    }
}

Graph::Graph(std::shared_ptr<Graph> graph, uint16_t stream_id)
        : m_context(graph->m_context)
        , m_program(graph->m_program)
        , m_networkName(graph->m_networkName)
        , m_config(graph->m_config)
        , m_stream_id(stream_id)
        , m_state(0) {
    Build();
}

void Graph::UpdateLayersMaps() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::UpdateLayersMaps");
    primitiveIDs = m_program->primitive_ids;
    prevPrimitiveIDs = m_program->prevPrimitiveIDs;
    profilingIDs = m_program->profiling_ids;
    perfMap = m_program->perfMap;
    outputDims = m_program->outputDims;
}

void Graph::Build() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::Build");
    UpdateLayersMaps();

    if (GetMaxDynamicBatchSize() > 1) {
        int m_bv_sz = m_program->GetMaxBatchSizeForSingleProgram();
        for (int b = m_bv_sz - 1; b >= 0; b--) {
            auto network = BuildNetwork(m_program->GetCompiledProgram(b));
            m_networks.insert(m_networks.begin(), network);
        }
    } else {
        auto network = BuildNetwork(m_program->GetCompiledProgram());
        m_networks.emplace_back(network);
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dry_run_path.empty()) {
        CNNNetwork net(GetExecGraphInfo());
        net.serialize(debug_config->dry_run_path);
        exit(0);
    }


    GPU_DEBUG_IF(!debug_config->dump_graphs.empty() && m_stream_id == 0) {
        static int net_id = 0;
        auto steps_info = GetNetwork()->get_optimizer_passes_info();
        size_t step_idx = 0;
        for (auto& step : steps_info) {
            CNNNetwork net(GetExecGraphInfoByPrimitivesInfo(step.second, true));
            net.serialize(debug_config->dump_graphs + std::to_string(net_id) + "_" +
                          std::to_string(step_idx) + "_" + step.first + "_graph.xml");
            step_idx++;
        }
        net_id++;
    }
}

bool Graph::use_external_queue() const {
    return m_context->get_external_queue() != nullptr;
}

std::shared_ptr<cldnn::network> Graph::BuildNetwork(std::shared_ptr<cldnn::program> program) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::BuildNetwork");
    std::shared_ptr<cldnn::network> network = nullptr;

    auto externalQueue = m_context->get_external_queue();
    if (externalQueue) {
        if (m_config.get_property(ov::num_streams) != 1)
            IE_THROW(ParameterMismatch) << "Throughput streams can't be used with shared queue!\n";
        auto &engine = m_program->get_engine();
        network = std::make_shared<cldnn::network>(program, engine.create_stream(m_config, externalQueue), m_stream_id);
    } else {
        network = std::make_shared<cldnn::network>(program, m_stream_id);
    }

    return network;
}

Graph::variable_states_map Graph::AllocateVariablesMemories() {
    Graph::variable_states_map states {};
    const auto& memStatesInfo = m_program->GetVariablesStatesInfo();
    OPENVINO_ASSERT(memStatesInfo.empty() || !GetNetwork()->is_dynamic(), "[GPU] Dynamic shapes are not supported yet for stateful models");
    for (const auto& memStateInfo : memStatesInfo) {
        std::vector<cldnn::layout> orderedLayouts {memStateInfo.second.begin(), memStateInfo.second.end()};
        std::sort(orderedLayouts.begin(), orderedLayouts.end(), [](cldnn::layout& first, cldnn::layout& second) {
            return first.batch() < second.batch();
        });
        std::vector<cldnn::network::VariableState::Ptr> memoryStates;
        memoryStates.reserve(orderedLayouts.size());
        for (const auto& layout : orderedLayouts)
            memoryStates.push_back(std::make_shared<cldnn::network::VariableState>(get_engine().allocate_memory(layout, false)));
        states.insert({memStateInfo.first, memoryStates });
    }
    return states;
}

std::shared_ptr<ngraph::Function> Graph::GetExecGraphInfoByPrimitivesInfo(std::vector<cldnn::primitive_info>& primitives_info,
                                                                          bool filter_const_primitives) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::GetExecGraphInfoByPrimitivesInfo");
    if (m_config.get_property(ov::enable_profiling)) {
        try {
            // Update may throw an exception for step-by-step runtime graph dump,
            // since network->get_executed_primitives() method can't be called before network execution
            UpdatePerfStatistics();
        } catch (std::exception&) {
        }
    }

    std::map<std::string, std::shared_ptr<ngraph::Node>> node2layer;

    ngraph::ResultVector results;
    ngraph::ParameterVector params;
    ngraph::NodeVector nodes;

    auto data_type_to_precision = [](cldnn::data_types dt) {
        switch (dt) {
            case cldnn::data_types::bin: return Precision::BIN;
            case cldnn::data_types::f32: return Precision::FP32;
            case cldnn::data_types::f16: return Precision::FP16;
            case cldnn::data_types::i32: return Precision::I32;
            case cldnn::data_types::i64: return Precision::I64;
            case cldnn::data_types::u8: return Precision::U8;
            case cldnn::data_types::i8: return Precision::I8;
            default: return Precision::UNSPECIFIED;
        }
    };

    // TODO: Adjust output layer names to be aligned with ngraph and add new ops
    auto to_IE_type_name = [](const std::string& cldnn_name) -> std::string{
        static std::map<std::string, std::string> type_n2l {
                { "activation", "Activation" },
                { "arg_max_min", "ArgMax" },
                { "batch_norm", "BatchNormalization" },
                { "binary_convolution", "BinaryConvolution" },
                { "border", "Pad" },
                { "concatenation", "Concat" },
                { "convolution", "Convolution" },
                { "deformable_convolution", "DeformableConvolution" },
                { "crop", "Crop" },
                { "custom_gpu_primitive", "CustomGPUPrimitive" },
                { "data", "Const" },
                { "deconvolution", "Deconvolution" },
                { "depth_to_space", "DepthToSpace" },
                { "detection_output", "DetectionOutput" },
                { "eltwise", "Eltwise" },
                { "fully_connected", "FullyConnected" },
                { "gather", "Gather" },
                { "gemm", "Gemm" },
                { "input_layout", "Input" },
                { "lrn", "LRN" },
                { "lstm", "LSTM" },
                { "lstm_elt", "LSTM_Eltwise" },
                { "lstm_gemm", "LSTM_Gemm" },
                { "mvn", "MVN" },
                { "normalize", "Normalize" },
                { "permute", "Permute" },
                { "pooling", "Pooling" },
                { "prior_box", "PriorBox" },
                { "proposal", "Proposal" },
                { "quantize", "Quantize" },
                { "region_yolo", "RegionYolo" },
                { "reorder", "Reorder" },
                { "reorg_yolo", "ReorgYolo" },
                { "reshape", "Reshape" },
                { "reverse_sequence", "ReverseSequence" },
                { "roi_pooling", "ROIPooling" },
                { "scale", "ScaleShift" },
                { "shuffle_channels", "ShuffleChannels" },
                { "softmax", "SoftMax" },
                { "split", "Split" },
                { "strided_slice", "StridedSlice" },
                { "tile", "Tile" },
                { "resample", "Resample" },
                { "interp", "Interp" },
                { "reduce", "Reduce" },
                { "reduce_max", "ReduceMax" },
                { "reduce_min", "ReduceMin" },
                { "reduce_mean", "ReduceMean" },
                { "reduce_prod", "ReduceProd" },
                { "reduce_sum", "ReduceSum" },
                { "reduce_and", "ReduceAnd" },
                { "reduce_or", "ReduceOr" },
                { "reduce_sum_square", "ReduceSumSquare" },
                { "reduce_l1", "ReduceL1" },
                { "reduce_l2", "ReduceL2" },
                { "reduce_log_sum", "ReduceLogSum" },
                { "reduce_log_sum_exp", "ReduceLogSumExp" },
                { "space_to_depth", "SpaceToDepth" },
        };

        if (type_n2l.find(cldnn_name) != type_n2l.end())
            return type_n2l.at(cldnn_name);

        return cldnn_name;
    };

    auto concat_strings = [](std::vector<std::string> strs, char sep) -> std::string {
        if (strs.empty())
            return "";

        std::string res = strs[0];
        for (size_t i = 1; i < strs.size(); i++) {
            res += sep + strs[i];
        }

        return res;
    };

    auto remove_type_from_name = [](const std::string& name) -> std::string {
        auto it = std::find(name.begin(), name.end(), ':');
        if (it == name.end() || (it + 1) == name.end())
            return name;

        return std::string((it+1), name.end());
    };

    auto extIdMap = GetNetwork()->get_ext_id_mapping();

    auto find_origin_layers = [&](const std::string& name) -> std::vector<std::string> {
        if (extIdMap.find(name) == extIdMap.end()) {
            return {};
        }
        return { extIdMap.at(name) };
    };

    auto get_inputs = [&] (const cldnn::primitive_info& prim_info) {
        ngraph::OutputVector inputs;

        auto& deps = prim_info.c_dependencies;

        // Decrease expected dependencies count if there is a const input without original id in the IR
        for (auto& dep : deps) {
            auto dep_it = std::find_if(primitives_info.begin(), primitives_info.end(), [&](cldnn::primitive_info& entry) {
                return entry.original_id == dep;
            });

            if (filter_const_primitives) {
                if (dep_it == primitives_info.end())
                    continue;
                if (dep_it->type_id == "data") {
                    continue;
                }
            }
            auto node_it = node2layer.find(dep);
            if (node_it != node2layer.end()) {
                inputs.push_back(node_it->second->get_default_output());
            }
        }

        return inputs;
    };

    auto create_ngraph_node = [&](const cldnn::primitive_info& prim_info) {
        const auto& user_ids = prim_info.c_users;
        size_t output_size = user_ids.size();
        bool is_output = user_ids.empty();
        auto out_et = cldnn::data_type_to_element_type(prim_info.output_layout.data_type);
        auto out_pshape = prim_info.output_layout.get_partial_shape();
        std::shared_ptr<ngraph::Node> return_node;

        if (prim_info.type_id == "input_layout") {
            auto param = std::make_shared<ngraph::op::Parameter>(out_et, out_pshape);
            params.push_back(param);
            return_node = param;
        } else {
            return_node = std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(get_inputs(prim_info), output_size);

            if (is_output) {    // create additional result node
                nodes.push_back(return_node);
                node2layer[prim_info.original_id] = return_node;
                return_node->set_output_type(0, out_et, out_pshape);
                results.emplace_back(std::make_shared<ngraph::op::Result>(return_node->get_default_output()));
            } else {
                size_t port = 0;
                for (auto& usr_id : user_ids) {
                    auto usr_it = std::find_if(primitives_info.begin(), primitives_info.end(), [&](cldnn::primitive_info& entry) {
                        return entry.original_id == usr_id;
                    });
                    if (usr_it == primitives_info.end())
                        continue;

                    return_node->set_output_type(port, out_et, out_pshape);
                    port++;
                }
            }
        }

        auto layerName = remove_type_from_name(prim_info.original_id);
        return_node->set_friendly_name(layerName);
        if (is_output)
            results.back()->set_friendly_name(layerName + "_result");

        std::map<std::string, std::string> info;
        Precision prec = data_type_to_precision(prim_info.output_layout.data_type);
        Precision inference_precision = data_type_to_precision(prim_info.runtime_precision);
        info[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = prec.name();
        info[ExecGraphInfoSerialization::LAYER_TYPE] = to_IE_type_name(prim_info.type_id);
        info[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = prim_info.layout_str;
        info[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(prim_info.exec_id);
        info[ExecGraphInfoSerialization::IMPL_TYPE] = prim_info.kernel_id;
        info[ExecGraphInfoSerialization::RUNTIME_PRECISION] = inference_precision.name();

        std::vector<std::string> originalNames{find_origin_layers(prim_info.original_id)};
        for (auto& fused_id : prim_info.c_fused_ids) {
            for (auto& origin_id : find_origin_layers(fused_id)) {
                if (std::find(originalNames.begin(), originalNames.end(), origin_id) == originalNames.end())
                    originalNames.push_back(origin_id);
            }
        }
        info[ExecGraphInfoSerialization::ORIGINAL_NAMES] = concat_strings(originalNames, ',');

        std::string exec_time = "not_executed";
        if (perfMap.find(prim_info.original_id) != perfMap.end()) {
            auto perfCounter = perfMap.at(prim_info.original_id).second;
            if (perfCounter.num > 0) {
                exec_time = std::to_string(perfCounter.realTime_avg());
            }
        }
        info[ExecGraphInfoSerialization::PERF_COUNTER] = exec_time;

        for (auto&& kvp : info) {
            return_node->get_rt_info()[kvp.first] = kvp.second;
            if (is_output)
                results.back()->get_rt_info()[kvp.first] = kvp.second;
        }
        if (is_output)
            results.back()->get_rt_info()[ExecGraphInfoSerialization::LAYER_TYPE] = "Result";

        nodes.push_back(return_node);
        node2layer[prim_info.original_id] = return_node;
    };

    if (filter_const_primitives) {
        for (auto& pi : primitives_info) {
            // extract mutable_data primitives and connect it's dependencies and users directly
            if (pi.type_id == "mutable_data") {
                if (pi.c_dependencies.size() == 1 && !pi.c_users.empty()) {
                    auto dep = pi.c_dependencies[0];
                    auto users = pi.c_users;
                    auto it = std::find_if(primitives_info.begin(), primitives_info.end(), [&](cldnn::primitive_info& entry) {
                        return entry.original_id == dep;
                    });
                    if (it == primitives_info.end())
                        continue;

                    auto& dep_users = it->c_users;
                    // Remove mutable data from users list
                    dep_users.erase(std::find_if(dep_users.begin(), dep_users.end(), [&](std::string user_id) {
                        return user_id == pi.original_id;
                    }));

                    // Add mutable data users to it's dependency users
                    dep_users.insert(dep_users.end(), users.begin(), users.end());

                    for (auto& user : users) {
                        it = std::find_if(primitives_info.begin(), primitives_info.end(), [&](cldnn::primitive_info& entry) {
                            return entry.original_id == user;
                        });
                        if (it == primitives_info.end())
                            continue;

                        for (auto& d : it->c_dependencies) {
                            if (d == pi.original_id)
                                d = dep;
                        }
                    }
                }
            }
        }
    }

    for (auto& pi : primitives_info) {
        if (filter_const_primitives) {
            // Skip const inputs
            if (pi.type_id == "data") {
                continue;
            }

            // Skip mutable_data
            if (pi.type_id == "mutable_data" &&
                pi.c_dependencies.size() == 1 &&
                !pi.c_users.empty()) {
                continue;
            }
        }

        create_ngraph_node(pi);
    }

    return std::make_shared<ngraph::Function>(results, params, "runtime_gpu_graph");
}

// Cache blob format:
//     [ ov::intel_gpu::Program::inputLayouts ]
//     [ ov::intel_gpu::Graph::primitiveIDs ]
//     [ ov::intel_gpu::Graph::outputDims ]
//     [ cldnn::network ]
void Graph::Export(cldnn::BinaryOutputBuffer &ob) {
    bool need_onednn_engine = false;
#ifdef ENABLE_ONEDNN_FOR_GPU
    try {
        get_engine().get_onednn_engine();
        need_onednn_engine = true;
    } catch (ov::AssertFailure &) {
        need_onednn_engine = false;
    }
#endif  // ENABLE_ONEDNN_FOR_GPU
    ob << need_onednn_engine;

    ob << m_program->inputLayouts;
    ob << m_program->GetVariablesStatesInfo();
    ob << primitiveIDs;
    ob << prevPrimitiveIDs;
    ob << profilingIDs;
    {
        ob << perfMap.size();
        for (auto& perf_item : perfMap) {
            ob << perf_item.first;
            ob << perf_item.second.second.layerType;
            ob << cldnn::make_data(&perf_item.second.second.status, sizeof(InferenceEngine::InferenceEngineProfileInfo::LayerStatus));
            ob << perf_item.second.second.isCPU;
            ob << perf_item.second.second.parentPrimitive;
        }
    }
    ob << outputDims;

    ob << m_networks.size();
    for (auto net : m_networks) {
        net->save(ob);
    }
}

std::shared_ptr<ngraph::Function> Graph::GetExecGraphInfo() {
    auto primitives_info = GetNetwork()->get_primitives_info();
    return GetExecGraphInfoByPrimitivesInfo(primitives_info, true);
}


void Graph::UpdatePerfStatistics() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::UpdatePerfStatistics");
    if (GetNetworksCount() == 0) {
        return;
    }

    // Collect timings
    auto collectTimings = [](cldnn::instrumentation::profiling_info& cldnnInfo, PerfCounter& pc) {
        for (auto &interval : cldnnInfo.intervals) {
            using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
            auto count = std::chrono::duration_cast<duration_t>(interval.value->value()).count();

            if (interval.stage == cldnn::instrumentation::profiling_stage::submission) {
                pc.cpu_uSec += count;
            } else if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                pc.realTime_uSec += count;
            } else if (interval.stage == cldnn::instrumentation::profiling_stage::duration) {  // "duration" is used for CPU layers
                pc.cpu_uSec += count;

                if (pc.num == 0)
                    pc.isCPU = true;
            }
        }
    };

    std::map<cldnn::primitive_id, cldnn::event::ptr> executedPrimitives = GetNetwork()->get_executed_primitives();
    auto allPrimitives = GetNetwork()->get_all_primitives();

    // Get profiling info for all layers
    for (auto &profiledID : profilingIDs) {
        auto pcIter = perfMap.find(profiledID);

        if (pcIter == perfMap.end())  continue;

        auto execIter = executedPrimitives.find(profiledID);
        auto& perfCount = pcIter->second.second;
        // Change status if layer wasn't executed by cldnn engine
        if (execIter == executedPrimitives.end()) {
            if (perfCount.num == 0) {
                perfCount.status = InferenceEngineProfileInfo::OPTIMIZED_OUT;
            }
            continue;
        }

        auto event = execIter->second;
        executedPrimitives.erase(execIter);

        cldnn::instrumentation::profiling_info cldnnInfo{profiledID, event->get_profiling_info()};

        collectTimings(cldnnInfo, perfCount);
        perfCount.num++;
    }

    for (auto &executedID : executedPrimitives) {
        auto pcIter = perfMap.find(executedID.first);
        if (pcIter == perfMap.end()) {
            perfMap[executedID.first].first = executedID.first;
            pcIter = perfMap.find(executedID.first);
            auto& perfCount = pcIter->second.second;

            cldnn::instrumentation::profiling_info cldnnInfo{executedID.first, executedID.second->get_profiling_info()};

            collectTimings(cldnnInfo, perfCount);
            perfCount.num++;
        }
    }
}

bool Graph::IsLoaded() const {
    return GetNetwork() != nullptr;
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> Graph::GetPerformanceCounts() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::GetPerformanceCounts");
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> result;
    bool combinePrimByIRLayers = false;
    unsigned i = 0;
    auto allIds = GetNetwork()->get_all_primitive_org_ids();
    auto executedPrimitives = GetNetwork()->get_executed_primitives();
    auto primitivesInfo = GetNetwork()->get_primitives_info();
    auto extIdMap = GetNetwork()->get_ext_id_mapping();
    std::map<std::string, std::string> implementation_info;

    if (GetNetwork()->get_program() == nullptr) {
        for (auto& pi : primitivesInfo) {
            implementation_info[pi.original_id] = pi.kernel_id;
        }
    }

    auto getUpperCaseName = [](std::string name) {
        std::vector<char> res;
        bool convert_next_to_upper = true;
        for (size_t i = 0; i < name.length(); i++) {
            char c = convert_next_to_upper ? toupper(name[i]) : name[i];
            if (c == '_') {
                convert_next_to_upper = true;
            } else {
                convert_next_to_upper = false;
                res.push_back(c);
            }
        }

        return std::string(res.begin(), res.end());
    };

    auto getClearName = [](std::string name) {
        if (name.find(":") != std::string::npos) {
            name = name.substr(name.find(":") + 1, name.length());
        }
        return name;
    };

    auto getFromProfiling = [&](std::string primId) -> bool {
        auto perfIter = perfMap.find(primId);

        if (perfIter == perfMap.end())
            return false;

        auto layerName = getClearName(perfIter->second.first);

        const auto& perfCounter = perfIter->second.second;

        if (!perfCounter.parentPrimitive.empty() && combinePrimByIRLayers)
            return false;

        auto& extPerfEntry = result[layerName];

        memset(extPerfEntry.exec_type, 0, sizeof(extPerfEntry.exec_type));
        if (perfCounter.isCPU) {
            static const std::string cpuExecType("CPU");
            cpuExecType.copy(extPerfEntry.exec_type, cpuExecType.length());  // Override execType as CPU
        } else {
            std::string impl;
            if (GetNetwork()->get_program() != nullptr) {
                impl = GetNetwork()->get_implementation_info(primId);
            } else {
                if (implementation_info.find(primId) != implementation_info.end()) {
                    impl = implementation_info[primId];
                } else {
                    impl = "undef";
                }
            }
            impl.copy(extPerfEntry.exec_type, impl.length());
        }

        extPerfEntry.execution_index = i++;
        extPerfEntry.status = perfCounter.status;
        extPerfEntry.cpu_uSec = perfCounter.cpu_avg();
        extPerfEntry.realTime_uSec = perfCounter.realTime_avg();

        if (combinePrimByIRLayers) {
            std::string kernelId = "";
            long long kernelTime = 0;  // used for finding the most complex computation kernel in sub_graph for perf stat
            for (auto &id : profilingIDs) {
                auto iter = perfMap.find(id);
                if (iter == perfMap.end())  continue;

                const auto &pc = iter->second.second;
                if (id != primId && pc.parentPrimitive == primId) {
                    extPerfEntry.cpu_uSec += pc.cpu_avg();
                    extPerfEntry.realTime_uSec += pc.realTime_avg();
                    if (pc.realTime_avg() > kernelTime) {
                        kernelTime = pc.realTime_avg();
                        kernelId = id;
                    }
                    allIds.erase(std::find(allIds.begin(), allIds.end(), id));
                }
            }
            if (!kernelId.empty()) {
                std::string impl_info = GetNetwork()->get_implementation_info(kernelId);
                std::memcpy(extPerfEntry.exec_type, &impl_info[0], impl_info.length());
            }
        }

        getUpperCaseName(perfCounter.layerType).copy(extPerfEntry.layer_type, perfCounter.layerType.length());
        return true;
    };

    // Step 1. Get all primitives in execution order which was added by GPU plugin
    for (auto& primId : profilingIDs) {
        getFromProfiling(primId);
    }

    // Step 2. Find all other primitives which was added while optimization process and executed after
    for (auto& primId : allIds) {
        auto perfIter = perfMap.find(primId);
        if (perfIter == perfMap.end())  continue;

        bool existInProfiling = std::find(profilingIDs.begin(), profilingIDs.end(), primId) != profilingIDs.end();
        if ((!existInProfiling || (existInProfiling && perfIter->second.first.length() == 0)) &&
            executedPrimitives.find(primId) != executedPrimitives.end()) {
            auto event = executedPrimitives.at(primId);

            cldnn::instrumentation::profiling_info cldnnInfo{primId, event->get_profiling_info()};

            // Collect timings
            long long cpuTime = 0;
            long long deviceTime = 0;

            for (auto &interval : cldnnInfo.intervals) {
                using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
                auto count = std::chrono::duration_cast<duration_t>(interval.value->value()).count();

                if (interval.stage == cldnn::instrumentation::profiling_stage::submission) {
                    cpuTime += count;
                } else if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                    deviceTime += count;
                } else if (interval.stage == cldnn::instrumentation::profiling_stage::duration) {  // "duration" is used for CPU layers
                    cpuTime += count;
                }
            }

            std::string layerName = getClearName(primId);

            for (auto& pi : primitivesInfo) {
                if (pi.original_id == primId) {
                    if (pi.type_id == "mutable_data")
                        continue;

                    auto& extPerfEntry = result[layerName];

                    if (pi.is_cpu) {
                        static const std::string cpuExecType("CPU");
                        memset(extPerfEntry.exec_type, 0, sizeof(extPerfEntry.exec_type));
                        cpuExecType.copy(extPerfEntry.exec_type, cpuExecType.length());  // Override execType as CPU
                    } else {
                        std::string impl = pi.kernel_id;
                        impl.copy(extPerfEntry.exec_type, impl.length());
                    }

                    getUpperCaseName(pi.type_id).copy(extPerfEntry.layer_type, pi.type_id.length());
                    extPerfEntry.execution_index = i++;
                    extPerfEntry.status = InferenceEngineProfileInfo::LayerStatus::EXECUTED;
                    extPerfEntry.cpu_uSec = cpuTime;
                    extPerfEntry.realTime_uSec = deviceTime;

                    if (pi.type_id == "input_layout") {
                        const std::string input_string = "Input";
                        const std::string undef_string = "undef";
                        input_string.copy(extPerfEntry.layer_type, 256);
                        undef_string.copy(extPerfEntry.exec_type, 256);
                    }
                }
            }
        }
    }

    // Step 3. Checking primitives which has been deleted from execution order but added by GPU plugin
    for (auto& primId : profilingIDs) {
        if (std::find(allIds.begin(), allIds.end(), primId) == allIds.end()) {
            getFromProfiling(primId);
        }
    }

    for (auto& p : extIdMap) {
        if (p.first.find(p.second) != std::string::npos) {
            continue;
        }
        auto first_res = result.find(getClearName(p.first));
        auto second_res = result.find(getClearName(p.second));

        if (first_res != result.end() && second_res != result.end() && first_res != second_res) {
            std::swap(first_res->second.cpu_uSec,        second_res->second.cpu_uSec);
            std::swap(first_res->second.realTime_uSec,   second_res->second.realTime_uSec);
            std::swap(first_res->second.status,          second_res->second.status);
            std::swap(first_res->second.exec_type,       second_res->second.exec_type);
            std::swap(first_res->second.execution_index, second_res->second.execution_index);
        }
    }
    return result;
}

std::shared_ptr<cldnn::network> Graph::GetNetwork(size_t idx) const {
    if (idx >= GetNetworksCount())
        IE_THROW() << "Unable to find network with id=" << idx << ". Stored networks count: " << GetNetworksCount();

    return m_networks[idx];
}


std::string Graph::MapOutputName(std::string outName) const {
    auto networkOutputsIDs = GetNetwork()->get_output_ids();
    auto allPrimitiveIds = GetNetwork()->get_all_primitives();

    // Find correct output ID. Start with name stored in IR.
    if (primitiveIDs.find(outName) == primitiveIDs.end()) {
        IE_THROW() << "output with name " << outName << " was not found in primitiveIDs";
    }
    std::string outputID = primitiveIDs.at(outName);
    while (std::find(networkOutputsIDs.begin(), networkOutputsIDs.end(), outputID) == networkOutputsIDs.end()) {
        // If current ID isn't found in cldnn network outputs, get previous primitive id and try again.
        auto prim = allPrimitiveIds.find(outputID);
        if (prim == allPrimitiveIds.end()) {
            IE_THROW() << "Unknown primitive id " << outputID;
        }

        if (prevPrimitiveIDs.at(outputID).size() != 1 || prim->second != "_optimized_") {
            IE_THROW() << "Unable to find parent for output primitive " << outputID;
        }
        outputID = prevPrimitiveIDs.at(outputID)[0];
    }

    return outputID;
}

InferenceEngine::SizeVector Graph::GetOutputSize(std::string outName) const {
    auto res_output = outputDims.find(outName);

    InferenceEngine::SizeVector sz;
    if (res_output != outputDims.end())
        sz = res_output->second;
    else
        sz = outputDims.at(primitiveIDs.at(outName));

    return sz;
}

}  // namespace intel_gpu
}  // namespace ov
