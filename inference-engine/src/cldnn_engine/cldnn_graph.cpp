// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <list>
#include <set>
#include <unordered_set>
#include <sstream>
#include <api/cldnn.hpp>
#include <api/network.hpp>
#include <api/profiling.hpp>
#include <api/custom_gpu_primitive.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "cldnn_graph.h"
#include "simple_math.h"
#include <description_buffer.hpp>
#include <cldnn/cldnn_config.hpp>
#include <graph_tools.hpp>
#include <ie_layers_internal.hpp>
#include <net_pass.h>
#include "cldnn_infer_request.h"
#include <threading/ie_executor_manager.hpp>
#include "details/caseless.hpp"
#include <fstream>
#include <utility>
#include <sys/types.h>
#include <sys/stat.h>
#include <exec_graph_info.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

CLDNNGraph::CLDNNGraph(InferenceEngine::ICNNNetwork& network, gpu::ClContext::Ptr context, Config config, uint16_t stream_id)
    : m_context(context)
    , m_networkName(network.getName())
    , m_config(config)
    , m_stream_id(stream_id) {
    m_program = std::make_shared<Program>(network, GetEngine(), m_config);
    Build();
}

CLDNNGraph::CLDNNGraph(std::shared_ptr<CLDNNGraph> graph, uint16_t stream_id)
        : m_context(graph->m_context)
        , m_program(graph->m_program)
        , m_networkName(graph->m_networkName)
        , m_config(graph->m_config)
        , m_stream_id(stream_id) {
    Build();
}

void CLDNNGraph::UpdateLayersMaps() {
    primitiveIDs = m_program->primitiveIDs;
    primitivesToIRLayersMap = m_program->primitivesToIRLayersMap;
    IRToNgraphLayersMap = m_program->IRToNgraphLayersMap;
    prevPrimitiveIDs = m_program->prevPrimitiveIDs;
    profilingIDs = m_program->profilingIDs;
    perfMap = m_program->perfMap;
    outputDims = m_program->outputDims;
}

void CLDNNGraph::Build() {
    UpdateLayersMaps();

    if (GetMaxDynamicBatchSize() > 1) {
        int m_bv_sz = m_program->GetMaxBatchSizeForSingleProgram();
        for (int b = m_bv_sz - 1; b >= 0; b--) {
            auto network = BuildNetwork(m_program->getCompiledProgram(b));
            m_networks.insert(m_networks.begin(), network);
            GetEngine()->release_pending_memory(network->get_id());
        }
    } else {
        auto network = BuildNetwork(m_program->getCompiledProgram());
        m_networks.emplace_back(network);
        GetEngine()->release_pending_memory(network->get_id());
    }

    UpdateImplementationsMap();
}

std::shared_ptr<cldnn::network> CLDNNGraph::BuildNetwork(std::shared_ptr<cldnn::program> program) {
    auto network = std::make_shared<cldnn::network>(*program, m_stream_id);

    if (!m_config.graph_dumps_dir.empty() && m_stream_id == 0) {
        static int net_id = 0;
        auto steps_info = network->get_optimization_steps_info();
        size_t step_idx = 0;
        for (auto& step : steps_info) {
            CNNNetwork net(GetExecGraphInfoByPrimitivesInfo(step.second, true));
            net.serialize(m_config.graph_dumps_dir + std::to_string(net_id) + "_" +
                          std::to_string(step_idx) + "_" + step.first + "_graph.xml");
            step_idx++;
        }
        net_id++;
    }

    return network;
}

InferenceEngine::ICNNNetwork::Ptr CLDNNGraph::GetExecGraphInfoByPrimitivesInfo(std::vector<cldnn::primitive_info>& primitives_info,
                                                                               bool filter_const_primitives) {
    auto net = std::make_shared<details::CNNNetworkImpl>();
    net->setName("runtime_gpu_graph");
    if (m_config.useProfiling) {
        try {
            // Update may throw an exception for step-by-step runtime graph dump,
            // since network->get_executed_primitives() method can't be called before network execution
            UpdatePerfStatistics();
        } catch (std::exception&) {
        }
    }

    std::vector<std::pair<cldnn::primitive_info, CNNLayerPtr>> node2layer;

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

    auto to_IE_type_name = [](const std::string& cldnn_name) -> std::string{
        static std::map<std::string, std::string> type_n2l {
                { "activation", "Activation" },
                { "arg_max_min", "ArgMax" },
                { "average_unpooling", "AverageUnpooling" },
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

    auto split_string = [](std::string src, std::string delimiter = ",") -> std::vector<std::string> {
        std::vector<std::string> tokens;
        std::string tokenBuf;
        size_t prev = 0, pos = 0, srcLength = src.length(), delimLength = delimiter.length();
        do {
            pos = src.find(delimiter, prev);
            if (pos == std::string::npos) {
                pos = srcLength;
            }
            tokenBuf = src.substr(prev, pos - prev);
            if (!tokenBuf.empty()) {
                tokens.push_back(tokenBuf);
            }
            prev = pos + delimLength;
        } while (pos < srcLength && prev < srcLength);

        return tokens;
    };

    auto remove_type_from_name = [](const std::string& name) -> std::string {
        auto it = std::find(name.begin(), name.end(), ':');
        if (it == name.end() || (it + 1) == name.end())
            return name;

        return std::string((it+1), name.end());
    };

    auto find_origin_layers = [&](const std::string& name) -> std::vector<std::string> {
        if (primitivesToIRLayersMap.find(name) == primitivesToIRLayersMap.end())
            return {};

        auto cnn_names = primitivesToIRLayersMap.at(name);
        std::vector<std::string> res;

        for (auto& cnn_name : cnn_names) {
            if (IRToNgraphLayersMap.find(cnn_name) != IRToNgraphLayersMap.end()) {
                auto ngraph_names = split_string(IRToNgraphLayersMap.at(cnn_name));
                res.insert(res.end(), ngraph_names.begin(), ngraph_names.end());
            } else {
                res.push_back(cnn_name);
            }
        }
        return res;
    };

    auto create_layer = [&](const cldnn::primitive_info& prim_info) -> CNNLayer::Ptr {
        CNNLayer::Ptr layer(new CNNLayer({"name", "type", Precision::UNSPECIFIED}));

        layer->name = remove_type_from_name(prim_info.original_id);
        layer->type = to_IE_type_name(prim_info.type_id);
        layer->precision = data_type_to_precision(prim_info.output_layout.data_type);
        std::vector<std::string> originalNames{find_origin_layers(prim_info.original_id)};
        for (auto& fused_id : prim_info.c_fused_ids) {
            for (auto& origin_id : find_origin_layers(fused_id)) {
                if (std::find(originalNames.begin(), originalNames.end(), origin_id) == originalNames.end())
                    originalNames.push_back(origin_id);
            }
        }

        layer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES] = concat_strings(originalNames, ',');
        layer->params[ExecGraphInfoSerialization::IMPL_TYPE] = prim_info.kernel_id;
        layer->params[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = layer->precision.name();
        std::string exec_time = "not_executed";
        if (perfMap.find(prim_info.original_id) != perfMap.end()) {
            auto perfCounter = perfMap.at(prim_info.original_id).second;
            if (perfCounter.num > 0) {
                exec_time = std::to_string(perfCounter.realTime_avg());
            }
        }

        layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = exec_time;
        layer->params[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = prim_info.layout_str;
        layer->params[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(prim_info.exec_id);

        node2layer.emplace_back(prim_info, layer);

        size_t in_size = prim_info.c_dependencies.size();

        if (filter_const_primitives) {
            // Decrease expected dependencies count if there is a const input without original id in the IR
            for (auto& dep : prim_info.c_dependencies) {
                auto it = std::find_if(primitives_info.begin(), primitives_info.end(), [&](cldnn::primitive_info& entry) {
                    return entry.original_id == dep;
                });

                if (it == primitives_info.end())
                    --in_size;

                if (it->type_id == "data") {
                    std::vector<std::string> childOriginalNames{find_origin_layers(prim_info.original_id)};
                    --in_size;
                }
            }
        }
        layer->insData.resize(in_size);
        layer->outData.resize(prim_info.c_users.size());

        return layer;
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
        auto layer = create_layer(pi);
        net->addLayer(layer);
    }

    auto desc_from_layout = [&](cldnn::layout layout) -> TensorDesc {
        Precision precision = data_type_to_precision(layout.data_type);
        SizeVector dims;
        Layout l = Layout::NCHW;
        auto size = layout.size;
        if (layout.format.dimension() == 4) {
            dims = {static_cast<size_t>(size.batch[0]),
                    static_cast<size_t>(size.feature[0]),
                    static_cast<size_t>(size.spatial[1]),
                    static_cast<size_t>(size.spatial[0])};
        } else if (layout.format.dimension() == 5) {
            dims = {static_cast<size_t>(size.batch[0]),
                    static_cast<size_t>(size.feature[0]),
                    static_cast<size_t>(size.spatial[2]),
                    static_cast<size_t>(size.spatial[1]),
                    static_cast<size_t>(size.spatial[0])};
            l = Layout::NCDHW;
        } else if (layout.format.dimension() == 6) {
            dims = {static_cast<size_t>(size.batch[0]),
                    static_cast<size_t>(size.feature[0]),
                    static_cast<size_t>(size.spatial[3]),
                    static_cast<size_t>(size.spatial[2]),
                    static_cast<size_t>(size.spatial[1]),
                    static_cast<size_t>(size.spatial[0])};
            // Should be NC?DHW but there is no such layout yet
            l = Layout::BLOCKED;
        }
        TensorDesc dst{precision, dims, l};
        return dst;
    };

    for (auto& pair : node2layer) {
        auto pi = pair.first;
        auto layer = pair.second;
        auto user_ids = pi.c_users;
        for (int i = 0; i < user_ids.size(); i++) {
            auto it = std::find_if(node2layer.begin(), node2layer.end(), [&](std::pair<cldnn::primitive_info, CNNLayerPtr>& entry) {
                return entry.first.original_id == user_ids[i];
            });

            if (it == node2layer.end())
                continue;

            auto& child_layer = it->second;

            DataPtr data;
            if (i < layer->outData.size()) {
                std::string data_name = pi.original_id + "_out" + std::to_string(i);
                layer->outData[i] = std::make_shared<Data>(data_name, desc_from_layout(pi.output_layout));
                data = layer->outData[i];
                getCreatorLayer(data) = layer;
            } else {
                data = layer->outData[0];
            }

            int in_port_id = 0;
            for (auto& dep : it->first.c_dependencies) {
                if (filter_const_primitives) {
                    auto it = std::find_if(node2layer.begin(), node2layer.end(), [&](std::pair<cldnn::primitive_info, CNNLayerPtr>& entry) {
                        return entry.first.original_id == dep;
                    });

                    if (it == node2layer.end())
                        continue;
                }

                if (dep == pi.original_id && child_layer->insData[in_port_id].lock() == nullptr) {
                    getInputTo(data)[child_layer->name] = child_layer;
                    child_layer->insData[in_port_id] = data;
                    break;
                }
                in_port_id++;
            }
        }
    }
    // Specify inputs data
    for (auto& pair : node2layer) {
        auto pi = pair.first;
        auto layer = pair.second;
        if (pi.c_dependencies.size() != 0)
            continue;

        auto in_info = std::make_shared<InputInfo>();
        if (layer->outData.empty())
            continue;

        auto dt = layer->outData[0];
        auto tensor_desc = desc_from_layout(pi.output_layout);

        dt->setDims(tensor_desc.getDims());
        dt->setPrecision(tensor_desc.getPrecision());
        dt->setLayout(tensor_desc.getLayout());

        in_info->setInputData(dt);
        net->setInputInfo(in_info);
    }

    return net;
}

void CLDNNGraph::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    auto primitives_info = GetNetwork()->get_primitives_info();
    graphPtr = GetExecGraphInfoByPrimitivesInfo(primitives_info, true);
}


void CLDNNGraph::UpdatePerfStatistics() {
    if (GetNetworksCount() == 0) {
        return;
    }

    // Collect timings
    auto collectTimings = [](cldnn::instrumentation::profiling_info& cldnnInfo, PerfCounter& pc) {
        for (auto &interval : cldnnInfo.intervals) {
            using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
            auto count = std::chrono::duration_cast<duration_t>(interval.value->value()).count();

            if (interval.name == "submission") {
                pc.cpu_uSec += count;
            } else if (interval.name == "executing") {
                pc.realTime_uSec += count;
            } else if (interval.name == "duration") {  // "duration" is used for CPU layers
                pc.cpu_uSec += count;

                if (pc.num == 0)
                    pc.isCPU = true;
            }
        }
    };

    std::map<cldnn::primitive_id, cldnn::event> executedPrimitives = GetNetwork()->get_executed_primitives();
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

        cldnn::instrumentation::profiling_info cldnnInfo{profiledID, event.get_profiling_info()};

        collectTimings(cldnnInfo, perfCount);
        perfCount.num++;
    }

    for (auto &executedID : executedPrimitives) {
        auto pcIter = perfMap.find(executedID.first);
        if (pcIter == perfMap.end()) {
            perfMap[executedID.first].first = executedID.first;
            pcIter = perfMap.find(executedID.first);
            auto& perfCount = pcIter->second.second;

            cldnn::instrumentation::profiling_info cldnnInfo{executedID.first, executedID.second.get_profiling_info()};

            collectTimings(cldnnInfo, perfCount);
            perfCount.num++;
        }
    }
}

bool CLDNNGraph::IsLoaded() const {
    return GetNetwork() != nullptr;
}

void CLDNNGraph::UpdateImplementationsMap() {
    if (m_config.useProfiling) {
        auto extractImplementationFromInfo = [](const std::string& info) -> std::string {
            std::string def_implementation = "undef";
            std::string impl_section = "implementation :";
            std::string::size_type pos = info.find(impl_section);
            if (pos == std::string::npos) {
                return def_implementation;
            }

            std::string::size_type end_pos = info.find(',', pos);
            if (end_pos == std::string::npos) {
                return def_implementation;
            }

            std::string::size_type length = end_pos - pos - impl_section.size();

            auto trim = [](const std::string& str) {
                size_t first = str.find_first_not_of(' ');
                if (std::string::npos == first) {
                    return str;
                }
                size_t last = str.find_last_not_of(' ');
                return str.substr(first, (last - first + 1));
            };
            std::string tmp = trim(info.substr(pos + impl_section.size(), length));

            return tmp.length() > 1 ? tmp : def_implementation;
        };

        // Parse primitive info and extract implementation name.
        for (auto& id : profilingIDs) {
            std::string prim_info = "";
            try {
                prim_info = GetNetwork()->get_primitive_info(id);
            } catch (std::exception& /*e*/) { }

            implementationsMap.insert({id, extractImplementationFromInfo(prim_info)});
        }
    }
}

void CLDNNGraph::GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &result) const {
    bool combinePrimByIRLayers = false;
    unsigned i = 0;
    auto allIds = GetNetwork()->get_all_primitive_org_ids();
    auto executedPrimitives = GetNetwork()->get_executed_primitives();
    auto primitivesInfo = GetNetwork()->get_primitives_info();

    auto getUpperCaseName = [&](std::string name) {
        if (name.length() > 0)
            name[0] = toupper(name[0]);
        return name;
    };

    auto getFromProfiling = [&](std::string primId) -> bool {
        auto perfIter = perfMap.find(primId);

        if (perfIter == perfMap.end())  return false;

        const auto& layerName = perfIter->second.first;
        if (layerName.length() == 0)  // no layer directly associated
            return false;

        const auto& perfCounter = perfIter->second.second;

        if (!perfCounter.parentPrimitive.empty() && combinePrimByIRLayers)
            return false;

        auto& extPerfEntry = result[layerName];

        memset(extPerfEntry.exec_type, 0, sizeof(extPerfEntry.exec_type));
        if (perfCounter.isCPU) {
            static const std::string cpuExecType("CPU");
            cpuExecType.copy(extPerfEntry.exec_type, cpuExecType.length());  // Override execType as CPU
        } else {
            std::string impl = implementationsMap.at(primId);
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
            if (!kernelId.empty())
                implementationsMap.at(kernelId).copy(extPerfEntry.exec_type, implementationsMap.at(kernelId).length());
        }

        getUpperCaseName(perfCounter.layerType).copy(extPerfEntry.layer_type, perfCounter.layerType.length());
        return true;
    };

    // Step 1. Get all primitives in execution order which was added by clDNNPlugin
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

            cldnn::instrumentation::profiling_info cldnnInfo{primId, event.get_profiling_info()};

            // Collect timings
            long long cpuTime = 0;
            long long deviceTime = 0;

            for (auto &interval : cldnnInfo.intervals) {
                using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
                auto count = std::chrono::duration_cast<duration_t>(interval.value->value()).count();

                if (interval.name == "submission") {
                    cpuTime += count;
                } else if (interval.name == "executing") {
                    deviceTime += count;
                } else if (interval.name == "duration") {  // "duration" is used for CPU layers
                    cpuTime += count;
                }
            }

            std::string layerName = primId;
            if (primId.find(":") != std::string::npos) {
                layerName = primId.substr(primId.find(":") + 1, primId.length());
            }

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

    // Step 3. Checking primitives which has been deleted from execution order but added by clDNNPlugin
    for (auto& primId : profilingIDs)
        if (std::find(allIds.begin(), allIds.end(), primId) == allIds.end()) {
            getFromProfiling(primId);
        }
}

std::shared_ptr<cldnn::network> CLDNNGraph::GetNetwork(size_t idx) const {
    if (idx >= GetNetworksCount())
        THROW_IE_EXCEPTION << "Unable to find network with id=" << idx << ". Stored networks count: " << GetNetworksCount();

    return m_networks[idx];
}


std::string CLDNNGraph::MapOutputName(std::string outName) const {
    auto networkOutputsIDs = GetNetwork()->get_output_ids();
    auto allPrimitiveIds = GetNetwork()->get_all_primitives();

    // Find correct output ID. Start with name stored in IR.
    std::string outputID = primitiveIDs.at(outName);
    while (std::find(networkOutputsIDs.begin(), networkOutputsIDs.end(), outputID) == networkOutputsIDs.end()) {
        // If current ID isn't found in cldnn network outputs, get previous primitive id and try again.
        auto prim = allPrimitiveIds.find(outputID);
        if (prim == allPrimitiveIds.end()) {
            THROW_IE_EXCEPTION << "Unknown primitive id " << outputID;
        }

        if (prevPrimitiveIDs.at(outputID).size() != 1 || prim->second != "_optimized_") {
            THROW_IE_EXCEPTION << "Unable to find parent for output primitive " << outputID;
        }
        outputID = prevPrimitiveIDs.at(outputID)[0];
    }

    return outputID;
}

InferenceEngine::SizeVector CLDNNGraph::GetOutputSize(std::string outName) const {
    auto res_output = outputDims.find(outName);

    InferenceEngine::SizeVector sz;
    if (res_output != outputDims.end())
        sz = res_output->second;
    else
        sz = outputDims.at(primitiveIDs.at(outName));

    return sz;
}

};  // namespace CLDNNPlugin
