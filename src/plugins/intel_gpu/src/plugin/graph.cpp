// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/pass/serialize.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/profiling.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/simple_math.hpp"

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

namespace ov {
namespace intel_gpu {

Graph::Graph(std::shared_ptr<ov::Model> model, const RemoteContextImpl::Ptr& context, const ExecutionConfig& config, uint16_t stream_id)
    : m_context(context)
    , m_config(config)
    , m_stream_id(stream_id) {
    auto program_builder = std::make_shared<ProgramBuilder>(model, get_engine(), config, false);
    m_config = program_builder->get_config();

    build(program_builder->get_compiled_program());

    primitiveIDs = program_builder->primitive_ids;
    inputPrimitiveIDs = program_builder->inputPrimitiveIDs;
    prevPrimitiveIDs = program_builder->prevPrimitiveIDs;
    profilingIDs = program_builder->profiling_ids;
    perfMap = program_builder->perfMap;
    m_input_layouts = program_builder->get_input_layouts();
}

Graph::Graph(cldnn::BinaryInputBuffer &ib, const RemoteContextImpl::Ptr& context, const ExecutionConfig& config, uint16_t stream_id)
    : m_context(context)
    , m_config(config)
    , m_stream_id(stream_id) {
    bool need_onednn_engine = false;
    ib >> need_onednn_engine;
    if (need_onednn_engine) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        get_engine().create_onednn_engine(config);
#else
        OPENVINO_THROW("[GPU] Current model cache requires OneDNN, but cannot use it.");
#endif  // ENABLE_ONEDNN_FOR_GPU
    }

    ib >> m_input_layouts;
    ib >> primitiveIDs;
    ib >> inputPrimitiveIDs;
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
            ib >> cldnn::make_data(&perfEntry.status, sizeof(ov::ProfilingInfo::Status));
            perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
            ib >> perfEntry.isCPU;
            ib >> perfEntry.parentPrimitive;
        }
    }
    {
        bool bool_prop_value;
        ib >> bool_prop_value;
        m_config.set_property(ov::intel_gpu::partial_build_program(bool_prop_value));
        ib >> bool_prop_value;
        m_config.set_property(ov::intel_gpu::optimize_data(bool_prop_value));
        ib >> bool_prop_value;
        m_config.set_property(ov::intel_gpu::allow_new_shape_infer(bool_prop_value));
    }

    auto imported_prog = std::make_shared<cldnn::program>(get_engine(), m_config);
    imported_prog->load(ib);
    build(imported_prog);
}

Graph::Graph(std::shared_ptr<Graph> graph, uint16_t stream_id)
        : m_context(graph->m_context)
        , m_config(graph->m_config)
        , m_stream_id(stream_id)
        , primitiveIDs(graph->primitiveIDs)
        , inputPrimitiveIDs(graph->inputPrimitiveIDs)
        , prevPrimitiveIDs(graph->prevPrimitiveIDs)
        , perfMap(graph->perfMap)
        , profilingIDs(graph->profilingIDs)
        , m_input_layouts(graph->m_input_layouts) {
    build(graph->get_network()->get_program());
}

Graph::~Graph() {
    GPU_DEBUG_IF(cldnn::debug_configuration::get_instance()->host_time_profiling) {
        const auto log_level = cldnn::debug_configuration::get_instance()->host_time_profiling;

        auto get_time_str = [](int64_t time_mcs, int64_t iters_num = 1) {
            double time = static_cast<double>(time_mcs);
            time /= iters_num;

            std::stringstream ss;
            std::string resolution = " mcs";
            if (time > 1000.0) {
                resolution = " ms";
                time /= 1000.0;
            }
            ss << std::fixed << std::setprecision(2) << time << resolution;

            return ss.str();
        };

        auto print_entry = [this, &get_time_str, &log_level](std::string name, HostTimeProfilingEntry& entry, int64_t iters_num = 1) {
            if (log_level == 1) {
                GPU_DEBUG_COUT << "[stream_id=" << m_stream_id << "] " << name << " infer enqueue host time: "
                               << get_time_str(entry.enqueue, iters_num) << std::endl;
            } else if (log_level >= 2) {
                auto total_time = entry.inputs_processing + entry.enqueue + entry.wait + entry.outputs_processing;

                GPU_DEBUG_COUT << "[stream_id=" << m_stream_id << "] " << name << " infer host time: "
                               << get_time_str(total_time, iters_num) << std::endl;
                GPU_DEBUG_COUT << " - " << " Inputs processing: " << get_time_str(entry.inputs_processing, iters_num) << std::endl;
                GPU_DEBUG_COUT << " - " << " Enqueue: " << get_time_str(entry.enqueue, iters_num) << std::endl;
                GPU_DEBUG_COUT << " - " << " Wait: " << get_time_str(entry.wait, iters_num) << std::endl;
                GPU_DEBUG_COUT << " - " << " Outputs processing: " << get_time_str(entry.outputs_processing, iters_num) << std::endl;
            }
        };

        if (host_exec_times.size() >= 1) {
            print_entry("First", host_exec_times[0], 1);
        }

        if (host_exec_times.size() >= 2) {
            HostTimeProfilingEntry avg;

            const auto begin = std::begin(host_exec_times) + 1;
            const auto end = std::end(host_exec_times);
            avg.inputs_processing = std::accumulate(begin, end, 0,
                [](int64_t sum, const HostTimeProfilingEntry& entry) { return sum + entry.inputs_processing; });
            avg.enqueue = std::accumulate(begin, end, 0,
                [](int64_t sum, const HostTimeProfilingEntry& entry) { return sum + entry.enqueue; });
            avg.wait = std::accumulate(begin, end, 0,
                [](int64_t sum, const HostTimeProfilingEntry& entry) { return sum + entry.wait; });
            avg.outputs_processing = std::accumulate(begin, end, 0,
                [](int64_t sum, const HostTimeProfilingEntry& entry) { return sum + entry.outputs_processing; });

            const auto iters_num = host_exec_times.size() - 1;
            print_entry("Avg", avg, iters_num);
        }
    }
}

void Graph::build(std::shared_ptr<cldnn::program> program) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::build");

    auto external_queue = m_context->get_external_queue();
    if (external_queue) {
        OPENVINO_ASSERT(m_config.get_property(ov::num_streams) == 1, "[GPU] Throughput streams can't be used with shared queue!");
        const auto &engine = program->get_engine();
        m_network = std::make_shared<cldnn::network>(program, engine.create_stream(m_config, external_queue), m_stream_id);
    } else {
        m_network = std::make_shared<cldnn::network>(program, m_stream_id);
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dry_run_path.empty()) {
        ov::pass::Serialize(debug_config->dry_run_path, "").run_on_model(get_runtime_model());
        exit(0);
    }

    GPU_DEBUG_IF(!debug_config->dump_graphs.empty() && m_stream_id == 0) {
        static int net_id = 0;
        auto steps_info = get_network()->get_optimizer_passes_info();
        size_t step_idx = 0;
        for (auto& step : steps_info) {
            auto xml_path = debug_config->dump_graphs + std::to_string(net_id) + "_" + std::to_string(step_idx) + "_" + step.first + "_graph.xml";
            ov::pass::Serialize(xml_path, "").run_on_model(get_runtime_model(step.second, true));
            step_idx++;
        }
        net_id++;
    }
}

bool Graph::use_external_queue() const {
    return m_context->get_external_queue() != nullptr;
}

std::shared_ptr<ov::Model> Graph::get_runtime_model(std::vector<cldnn::primitive_info>& primitives_info, bool filter_const_primitives) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::get_runtime_model");
    if (m_config.get_property(ov::enable_profiling)) {
        try {
            // Update may throw an exception for step-by-step runtime graph dump,
            // since network->get_executed_primitives() method can't be called before network execution
            update_profiling_info();
        } catch (std::exception&) {
        }
    }

    std::map<std::string, std::shared_ptr<ov::Node>> node2layer;

    ov::ResultVector results;
    ov::ParameterVector params;
    ov::NodeVector nodes;

    // TODO: Adjust output layer names to be aligned with ov and add new ops
    auto to_OV_type_name = [](const std::string& cldnn_name) -> std::string{
        static std::map<std::string, std::string> type_n2l {
                { "activation", "Activation" },
                { "arg_max_min", "ArgMax" },
                { "batch_norm", "BatchNormalization" },
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
                { "lstm_elt", "LSTM_Eltwise" },
                { "mvn", "MVN" },
                { "normalize", "Normalize" },
                { "permute", "Permute" },
                { "pooling", "Pooling" },
                { "prior_box", "PriorBox" },
                { "proposal", "Proposal" },
                { "quantize", "Quantize" },
                { "region_yolo", "RegionYolo" },
                { "reorder", "Reorder" },
                { "rope", "RoPE" },
                { "reorg_yolo", "ReorgYolo" },
                { "reshape", "Reshape" },
                { "reverse_sequence", "ReverseSequence" },
                { "roi_pooling", "ROIPooling" },
                { "scale", "ScaleShift" },
                { "shuffle_channels", "ShuffleChannels" },
                { "softmax", "SoftMax" },
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

    auto extIdMap = get_network()->get_ext_id_mapping();

    auto find_origin_layers = [&](const std::string& name) -> std::vector<std::string> {
        if (extIdMap.find(name) == extIdMap.end()) {
            return {};
        }
        return { extIdMap.at(name) };
    };

    auto get_inputs = [&] (const cldnn::primitive_info& prim_info) {
        ov::OutputVector inputs;

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

    auto create_ov_node = [&](const cldnn::primitive_info& prim_info) {
        const auto& user_ids = prim_info.c_users;
        size_t output_size = user_ids.size();
        bool is_output = user_ids.empty();
        auto out_et = prim_info.output_layout.data_type;
        auto out_pshape = prim_info.output_layout.get_partial_shape();
        std::shared_ptr<ov::Node> return_node;

        if (prim_info.type_id == "input_layout") {
            auto param = std::make_shared<ov::op::v0::Parameter>(out_et, out_pshape);
            params.push_back(param);
            return_node = param;
            // create additional result node if parameter is output without post reorder
            if (is_output) {
                results.emplace_back(std::make_shared<ov::op::v0::Result>(return_node->get_default_output()));
            }
        } else {
            return_node = std::make_shared<ov::exec_model_info::ExecutionNode>(get_inputs(prim_info), output_size);

            if (is_output) {    // create additional result node
                nodes.push_back(return_node);
                node2layer[prim_info.original_id] = return_node;
                return_node->set_output_type(0, out_et, out_pshape);
                results.emplace_back(std::make_shared<ov::op::v0::Result>(return_node->get_default_output()));
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
        info[ov::exec_model_info::OUTPUT_PRECISIONS] = ov::element::Type(prim_info.output_layout.data_type).get_type_name();
        info[ov::exec_model_info::LAYER_TYPE] = to_OV_type_name(prim_info.type_id);
        info[ov::exec_model_info::OUTPUT_LAYOUTS] = prim_info.layout_str;
        info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(prim_info.exec_id);
        info[ov::exec_model_info::IMPL_TYPE] = prim_info.kernel_id;
        info[ov::exec_model_info::RUNTIME_PRECISION] = ov::element::Type(prim_info.runtime_precision).get_type_name();

        std::vector<std::string> originalNames{find_origin_layers(prim_info.original_id)};
        for (auto& fused_id : prim_info.c_fused_ids) {
            for (auto& origin_id : find_origin_layers(fused_id)) {
                if (std::find(originalNames.begin(), originalNames.end(), origin_id) == originalNames.end())
                    originalNames.push_back(origin_id);
            }
        }
        info[ov::exec_model_info::ORIGINAL_NAMES] = concat_strings(originalNames, ',');

        std::string exec_time = "not_executed";
        if (perfMap.find(prim_info.original_id) != perfMap.end()) {
            auto perfCounter = perfMap.at(prim_info.original_id).second;
            if (perfCounter.num > 0) {
                exec_time = std::to_string(perfCounter.realTime_avg());
            }
        }
        info[ov::exec_model_info::PERF_COUNTER] = exec_time;

        for (auto&& kvp : info) {
            return_node->get_rt_info()[kvp.first] = kvp.second;
            if (is_output)
                results.back()->get_rt_info()[kvp.first] = kvp.second;
        }
        if (is_output)
            results.back()->get_rt_info()[ov::exec_model_info::LAYER_TYPE] = "Result";

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

        create_ov_node(pi);
    }

    return std::make_shared<ov::Model>(results, params, "runtime_gpu_graph");
}

// Cache blob format:
//     [ ov::intel_gpu::ProgramBuilder::inputLayouts ]
//     [ ov::intel_gpu::Graph::primitiveIDs ]
//     [ cldnn::network ]
void Graph::export_model(cldnn::BinaryOutputBuffer &ob) {
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

    ob << m_input_layouts;
    ob << primitiveIDs;
    ob << inputPrimitiveIDs;
    ob << prevPrimitiveIDs;
    ob << profilingIDs;
    {
        ob << perfMap.size();
        for (auto& perf_item : perfMap) {
            ob << perf_item.first;
            ob << perf_item.second.second.layerType;
            ob << cldnn::make_data(&perf_item.second.second.status, sizeof(ov::ProfilingInfo::Status));
            ob << perf_item.second.second.isCPU;
            ob << perf_item.second.second.parentPrimitive;
        }
    }
    {
        ob << m_config.get_property(ov::intel_gpu::partial_build_program);
        ob << m_config.get_property(ov::intel_gpu::optimize_data);
        ob << m_config.get_property(ov::intel_gpu::allow_new_shape_infer);
    }

    ob.set_stream(m_network->get_stream_ptr().get());
    m_network->get_program()->save(ob);
}

std::shared_ptr<ov::Model> Graph::get_runtime_model() {
    auto primitives_info = get_network()->get_primitives_info();
    return get_runtime_model(primitives_info, true);
}


void Graph::update_profiling_info() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::update_profiling_info");
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

    std::map<cldnn::primitive_id, cldnn::event::ptr> executedPrimitives = get_network()->get_executed_primitives();

    // Get profiling info for all layers
    for (auto &profiledID : profilingIDs) {
        auto pcIter = perfMap.find(profiledID);

        if (pcIter == perfMap.end())  continue;

        auto execIter = executedPrimitives.find(profiledID);
        auto& perfCount = pcIter->second.second;
        // Change status if layer wasn't executed by cldnn engine
        if (execIter == executedPrimitives.end()) {
            if (perfCount.num == 0) {
                perfCount.status = ov::ProfilingInfo::Status::OPTIMIZED_OUT;
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

bool Graph::is_loaded() const {
    return get_network() != nullptr;
}

std::vector<ov::ProfilingInfo> Graph::get_profiling_info() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Graph::get_profiling_info");
    std::map<std::string, ov::ProfilingInfo> result;
    bool combinePrimByIRLayers = false;
    auto allIds = get_network()->get_all_primitive_org_ids();
    auto executedPrimitives = get_network()->get_executed_primitives();
    auto primitivesInfo = get_network()->get_primitives_info();
    auto extIdMap = get_network()->get_ext_id_mapping();
    std::map<std::string, std::string> implementation_info;

    if (get_network()->get_program() == nullptr) {
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

        if (perfCounter.isCPU) {
            extPerfEntry.exec_type = "CPU";
        } else {
            std::string impl;
            if (get_network()->get_program() != nullptr) {
                impl = get_network()->get_implementation_info(primId);
            } else {
                if (implementation_info.find(primId) != implementation_info.end()) {
                    impl = implementation_info[primId];
                } else {
                    impl = "undef";
                }
            }
            extPerfEntry.exec_type = impl;
        }

        extPerfEntry.status = perfCounter.status;
        extPerfEntry.cpu_time = std::chrono::microseconds(perfCounter.cpu_avg());
        extPerfEntry.real_time = std::chrono::microseconds(perfCounter.realTime_avg());
        extPerfEntry.node_name = layerName;

        if (combinePrimByIRLayers) {
            std::string kernelId = "";
            long long kernelTime = 0;  // used for finding the most complex computation kernel in sub_graph for perf stat
            for (auto &id : profilingIDs) {
                auto iter = perfMap.find(id);
                if (iter == perfMap.end())  continue;

                const auto &pc = iter->second.second;
                if (id != primId && pc.parentPrimitive == primId) {
                    extPerfEntry.cpu_time += std::chrono::microseconds(pc.cpu_avg());
                    extPerfEntry.real_time += std::chrono::microseconds(pc.realTime_avg());
                    if (pc.realTime_avg() > kernelTime) {
                        kernelTime = pc.realTime_avg();
                        kernelId = id;
                    }
                    allIds.erase(std::find(allIds.begin(), allIds.end(), id));
                }
            }
            if (!kernelId.empty()) {
                extPerfEntry.exec_type = get_network()->get_implementation_info(kernelId);
            }
        }

        extPerfEntry.node_type = getUpperCaseName(perfCounter.layerType);
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
                        extPerfEntry.exec_type = "CPU";
                    } else {
                        extPerfEntry.exec_type = pi.kernel_id;
                    }

                    extPerfEntry.node_type = getUpperCaseName(pi.type_id);
                    extPerfEntry.node_name = pi.original_id;
                    extPerfEntry.status = ov::ProfilingInfo::Status::EXECUTED;
                    extPerfEntry.cpu_time = std::chrono::microseconds(cpuTime);
                    extPerfEntry.real_time = std::chrono::microseconds(deviceTime);

                    if (pi.type_id == "input_layout") {
                        extPerfEntry.node_type = "Input";
                        extPerfEntry.exec_type = "undef";
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
            std::swap(first_res->second.cpu_time,        second_res->second.cpu_time);
            std::swap(first_res->second.real_time,   second_res->second.real_time);
            std::swap(first_res->second.status,          second_res->second.status);
            std::swap(first_res->second.exec_type,       second_res->second.exec_type);
        }
    }

    std::vector<ov::ProfilingInfo> res;
    for (auto& kv : result) {
        res.push_back(kv.second);
    }
    return res;
}

std::shared_ptr<cldnn::network> Graph::get_network() const {
    return m_network;
}

std::vector<cldnn::primitive_id> Graph::input_port_index_to_internal(size_t input_port_index) const {
    OPENVINO_ASSERT(inputPrimitiveIDs.count(input_port_index) != 0 && !inputPrimitiveIDs.at(input_port_index).empty(),
                    "[GPU] Internal name of input primitive not found at index ", input_port_index);
    return inputPrimitiveIDs.at(input_port_index);
}

std::string Graph::out_port_index_to_internal(size_t out_port_index) const {
    const auto& networkOutputsIDs = get_network()->get_output_ids();
    auto check_output = [&networkOutputsIDs](const cldnn::primitive_id& id) {
        return std::find(networkOutputsIDs.begin(), networkOutputsIDs.end(), id) != networkOutputsIDs.end();
    };

    OPENVINO_ASSERT(prevPrimitiveIDs.count(out_port_index) != 0,
                    "[GPU] Internal name of output primitive not found for index ", out_port_index);
    cldnn::primitive_id outputID = prevPrimitiveIDs.at(out_port_index);

    if (check_output(outputID)) {
        return outputID;
    }

    OPENVINO_ASSERT(primitiveIDs.find(outputID) != primitiveIDs.end(),
                    "[GPU] Output with name ", outputID, " was not found in primitiveIDs");
    outputID = primitiveIDs.at(outputID);

    if (check_output(outputID)) {
        return outputID;
    }

    OPENVINO_THROW("[GPU] Unable to map output port index ", out_port_index, " to the internal primitive id");
}

}  // namespace intel_gpu
}  // namespace ov
