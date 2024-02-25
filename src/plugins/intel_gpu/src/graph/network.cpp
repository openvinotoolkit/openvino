// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "openvino/util/file_util.hpp"

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"

#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "mutable_data_inst.h"
#include "condition_inst.h"
#include "loop_inst.h"
#include "assign_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"
#include "kv_cache_inst.h"
#include "program_helpers.h"
#include "to_string_utils.h"
#include "kernels_cache.hpp"
#include "program_dump_graph.h"

// TODO: Remove once we have an abstraction for kernels_cache
#include "kernel_base.h"

#include <algorithm>
#include <string>
#include <vector>
#include <stack>
#include <memory>
#include <set>
#include <utility>
#include <map>
#include <functional>
#include <fstream>

#ifdef GPU_DEBUG_CONFIG
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#endif

namespace cldnn {

namespace {

#ifdef GPU_DEBUG_CONFIG
void dump_perf_data_raw(std::string dump_path, const std::list<std::shared_ptr<primitive_inst>>& exec_order) {
    auto layouts_to_str = [](const std::vector<layout>& layouts) -> std::string {
        std::stringstream s;
        for (size_t i = 0; i < layouts.size(); i++) {
            s << layouts[i].to_short_string();
            if (i != layouts.size() - 1)
                s << ";";
        }
        return s.str();
    };

    const bool per_iter_mode = cldnn::debug_configuration::get_instance()->dump_profiling_data_per_iter != 0;
    const std::string perf_raw_csv_header = per_iter_mode ? "prim_id,prim_type,stage,net_in_shapes,in_shapes,out_shapes,impl,iter,time_usec\n"
                                                          : "prim_id,prim_type,stage,net_in_shapes,in_shapes,out_shapes,impl,iters,time_usec\n";
    std::ofstream of(dump_path);
    if (of.is_open()) {
        of << perf_raw_csv_header;
        for (auto& inst : exec_order) {
            auto prim_id = inst->id();
            auto& perf_data = inst->get_profiling_data();
            auto& perf_info = inst->get_profiling_info();
            std::vector<size_t> sorted_entries;
            std::transform(perf_data.begin(), perf_data.end(), std::back_inserter(sorted_entries),
            [](const std::pair<size_t, std::tuple<int64_t, size_t>>& e) {
                return e.first;
            });
            std::sort(sorted_entries.begin(), sorted_entries.end(), [&](size_t a, size_t b) -> bool {
                auto& a_info = perf_info.at(a);
                auto& b_info = perf_info.at(b);

                if (a_info.stage != b_info.stage) {
                    return static_cast<std::underlying_type<instrumentation::pipeline_stage>::type>(a_info.stage) <
                           static_cast<std::underlying_type<instrumentation::pipeline_stage>::type>(b_info.stage);
                }

                if (a_info.cache_hit != b_info.cache_hit)
                    return a_info.cache_hit;

                if (a_info.memalloc_info != b_info.memalloc_info)
                    return a_info.memalloc_info.length() < b_info.memalloc_info.length();

                size_t total_out_size_a = 0;
                size_t total_out_size_b = 0;
                for (auto& ol : a_info.output_layouts) {
                    total_out_size_a += ol.count();
                }
                for (auto& ol : b_info.output_layouts) {
                    total_out_size_b += ol.count();
                }
                return total_out_size_a < total_out_size_b;
            });
            for (auto& hash : sorted_entries) {
                auto& key = perf_info.at(hash);
                auto& entry = perf_data.at(hash);
                auto& time = std::get<0>(entry);
                auto num_iters = per_iter_mode ? key.iteration_num : std::get<1>(entry);
                int64_t time_avg = per_iter_mode ? time : time / num_iters;
                std::string net_in_l_str = layouts_to_str(key.network_input_layouts);
                std::string in_l_str = layouts_to_str(key.input_layouts);
                std::string out_l_str = layouts_to_str(key.output_layouts);
                std::string stage_suffix = "";
                if (key.cache_hit)
                    stage_suffix += " (cache_hit) ";
                if (key.memalloc_info != "")
                    stage_suffix += " (" + key.memalloc_info + ") ";
                of << prim_id << ","
                << inst->desc()->type_string() << ","
                << key.stage << stage_suffix << ","
                << net_in_l_str << ","
                << in_l_str << ","
                << out_l_str << ","
                << (key.stage == instrumentation::pipeline_stage::inference ? key.impl_name : "undef") << ","
                << num_iters << ","
                << time_avg << "\n";
            }
        }
    }
}

float convert_element(int64_t i) { return static_cast<float>(i); }
float convert_element(int32_t i) { return static_cast<float>(i); }

float convert_element(float f) { return f; }

float convert_element(ov::float16 h) { return static_cast<float>(h); }

size_t get_x_pitch(const layout& layout) {
    try {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless
        return 0;
    }
}

template <class T>
void dump(memory::ptr mem, stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto batch_size = std::max(std::min(debug_config->dump_layers_limit_batch, size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());
    std::stringstream buffer;

    if (!dump_raw) {
        for (cldnn::tensor::value_type g = 0; g < size.group[0]; ++g) {
            for (cldnn::tensor::value_type b = 0; b < batch_size; ++b) {
                for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f) {
                    for (cldnn::tensor::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (cldnn::tensor::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y) {
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = mem->get_layout().get_linear_offset(t);

                                for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[input_it]) << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < lock.size(); ++i) {
            buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
        }
    }
    file_stream << buffer.str();
}

void log_memory_to_file(memory::ptr mem, layout data_layout, stream& stream, std::string layerName, bool dump_raw) {
    std::cout << "Dump " << (dump_raw ? "raw " : "") << layerName << std::endl;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    std::string filename = debug_config->get_name_for_dump(layerName);
    filename = debug_config->dump_layers_path + filename + ".txt";
    std::ofstream file_stream(filename);
    if (!mem) {
        file_stream << "Empty" << std::endl;
        return;
    }

    // Reinterpret buffer to represent actual data layout
    auto actual_mem = mem->get_engine()->reinterpret_buffer(*mem, data_layout);

    auto mem_dt = actual_mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        dump<float>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::f16)
        dump<ov::float16>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i64)
        dump<int64_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i32)
        dump<int32_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i8)
        dump<int8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
}

void wait_for_the_turn() {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    bool need_to_wait;
    do {
        need_to_wait = false;
        struct stat buffer;
        for (auto pid : debug_config->after_proc) {
            auto path = "/proc/" + pid;
            std::cout << "check " + path << std::endl;
            if (stat(path.c_str(), &buffer) == 0) {
                need_to_wait = true;
                std::cout << "Being nice.. Wait for process " << pid << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
    } while (need_to_wait);
}

#else
void dump_perf_data_raw(std::string, const std::list<std::shared_ptr<primitive_inst>>&) {}
void log_memory_to_file(memory::ptr, layout, stream&, std::string, bool dump_raw) {}
void wait_for_the_turn() {}
#endif
}  // namespace

static uint32_t get_unique_net_id() {
    static std::atomic<uint32_t> id_gen{0};
    return ++id_gen;
}

static std::string get_file_path_for_binary_dump(cldnn::layout layout, std::string name) {
    std::string filename;
    std::string data_type = ov::element::Type(layout.data_type).get_type_name();
    std::string format = layout.format.to_string();
    std::string tensor;
    auto dims = layout.get_dims();
    for (size_t r = 0 ; r < layout.get_rank() ; r++) {
        tensor += ("_" + to_string(dims[r]));
    }

#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_GET_INSTANCE(debug_config);
    std::string layer_name = debug_config->get_name_for_dump(name);
    filename = debug_config->dump_layers_path + layer_name
                + "__" + data_type + "_" + tensor + "__" + format + ".bin";
#endif
    return filename;
}

/*
Network will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by propagate_constants
opt pass).
*/
network::network(program::ptr program, const ExecutionConfig& config, stream::ptr stream, bool is_internal, bool is_primary_stream)
    : _program(program)
    , _config(config)
    , _engine(program->get_engine())
    , _stream(stream)
    , _memory_pool(new memory_pool(program->get_engine()))
    , _internal(is_internal)
    , _is_primary_stream(is_primary_stream)
    , _enable_profiling(config.get_property(ov::enable_profiling))
    , _reset_arguments(true)
    , _shape_predictor(new ShapePredictor(&program->get_engine(), config.get_property(ov::intel_gpu::buffers_preallocation_ratio))) {
    if (!_internal) {
        net_id = get_unique_net_id();
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->after_proc.size() != 0) {
        wait_for_the_turn();
    }

    GPU_DEBUG_IF(debug_config->mem_preallocation_params.is_initialized) {
        auto& mem_preallocation_params = debug_config->mem_preallocation_params;
        _shape_predictor.reset(new ShapePredictor(&program->get_engine(),
                                                  mem_preallocation_params.next_iters_preallocation_count,
                                                  mem_preallocation_params.max_per_iter_size,
                                                  mem_preallocation_params.max_per_dim_diff,
                                                  mem_preallocation_params.buffers_preallocation_ratio));
    }

    calculate_weights_cache_capacity();
    allocate_primitives();
    configure_primitives_second_output();
    if (!_program->is_loaded_from_cache())
        check_names();
    build_insts_deps();
    build_exec_order();
    validate_primitives();
    add_default_output_chains();
}

network::network(engine& engine,
                 const topology& topo,
                 const ExecutionConfig& config,
                 bool is_internal,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor)
    : network(program::build_program(engine, topo, config, task_executor, is_internal), config, engine.create_stream(config), is_internal) {}

network::network(engine& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const ExecutionConfig& config,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                 bool is_internal)
    : network(program::build_program(engine, nodes, config, task_executor, is_internal), config, engine.create_stream(config), is_internal) {}

network::network(program::ptr program, uint16_t stream_id)
    : network(program, program->get_config(), program->get_engine().create_stream(program->get_config()), false, stream_id == 0) {}

network::network(program::ptr program, stream::ptr stream, uint16_t stream_id)
    : network(program, program->get_config(), stream, false, stream_id == 0) {}

network::~network() {
    if (_program != nullptr)
        _program->cancel_compilation_context();
    _memory_pool->clear_pool_for_network(net_id);
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
        dump_perf_data_raw(debug_config->dump_profiling_data + "/perf_raw" + std::to_string(net_id) + ".csv", _exec_order);
    }
}

network::ptr network::allocate_network(stream::ptr stream, program::ptr program, bool is_internal, bool is_primary_stream) {
    return std::make_shared<network>(program, program->get_config(), stream, is_internal, is_primary_stream);
}

network::ptr network::allocate_network(engine& engine, program::ptr program, bool is_internal, bool is_primary_stream) {
    auto stream = engine.create_stream(program->get_config());
    return std::make_shared<network>(program, program->get_config(), stream, is_internal, is_primary_stream);
}

network::ptr network::build_network(engine& engine,
                                    const topology& topology,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    bool is_internal) {
    return std::make_shared<network>(engine, topology, config, is_internal, task_executor);
}

network::ptr network::build_network(engine& engine,
                                    const std::set<std::shared_ptr<program_node>>& nodes,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    bool is_internal) {
    return std::make_shared<network>(engine, nodes, config, task_executor, is_internal);
}

void network::validate_primitives() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("validate_primitives");
    for (auto const& prim : _exec_order) {
        prim->validate();
    }
}

void network::set_arguments() {
    if (!_reset_arguments)
        return;

    for (auto const& prim : _exec_order) {
        if (!prim->is_dynamic()) {
            bool can_set_args = true;
            for (auto& dep : prim->dependencies()) {
                // Skip set args for nodes with dynamic & optimized_out dependency
                // This is needed to handle dynamic -> static cases like
                // (dynamic) -> reshape -> (static) -> some_op
                // In that case some_op is static and we may want to set arguments once,
                // but dynamic optimized out reshape means that output buffer of reshape is unavailable
                // and attempt to set args will fail.
                auto prim = dep.first->get_impl_params()->desc;
                if (dep.first->can_be_optimized() && (dep.first->is_dynamic() || prim->type == read_value::type_id()))
                    can_set_args = false;
            }

            if (can_set_args)
                prim->set_arguments();
        }
    }
    _reset_arguments = false;
}

void network::reset_execution(bool wait) {
    if (wait) {
        auto queue_type = get_config().get_property(ov::intel_gpu::queue_type);
        if (queue_type == QueueTypes::in_order) {
            get_stream().finish();
        } else if (queue_type == QueueTypes::out_of_order && _events.size() > 0) {
            std::vector<event::ptr> events;
            for (auto& pair : _events) {
                auto& ev = pair.second;
                if (ev->is_set())
                    continue;

                events.push_back(ev);
            }

            get_stream().wait_for_events(events);
        }
    }

    // Move events to temporarily map to deallocate them at the end of network::execute() call for better overlapping with
    // kernels execution, since it may take significant time for high amount of events
    _old_events = std::move(_events);
}

event::ptr network::set_input_data(const primitive_id& id, memory::ptr data) {
    GPU_DEBUG_TRACE_DETAIL << "Set input " << id << " " << data->get_layout().to_short_string() << std::endl;
    std::shared_ptr<primitive_inst> primitive_inst;

    primitive_inst = find_primitive(id);

    OPENVINO_ASSERT(primitive_inst != nullptr, "[GPU] topology doesn't contain primitive: ", id);

    if (primitive_inst->type() != input_layout::type_id()) {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }

    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);

    return input->set_data(data);
}

void network::add_default_output_chains() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("add_default_output_chains");
    for (auto& output : _outputs) {
        add_output_chain(output);
    }
}

void network::calculate_weights_cache_capacity() {
    auto get_buffer_size = [](const program_node& node) {
        size_t weights_size = 0;
        auto get_size = [](const layout& layout) {
            return layout.is_dynamic() ? 0 : layout.bytes_count();
        };

        #define is_weightable(T) node.is_type<T>() && node.as<T>().weights().is_constant()
        if (node.is_type<data>())
            weights_size = get_size(node.get_output_layout());
        else if (is_weightable(fully_connected))
            weights_size = get_size(node.as<fully_connected>().weights().get_output_layout());
        else if (is_weightable(convolution))
            weights_size = get_size(node.as<convolution>().weights().get_output_layout());
        else if (is_weightable(deconvolution))
            weights_size = get_size(node.as<deconvolution>().weights().get_output_layout());
        #undef is_weightable

        return weights_size;
    };

    size_t total_const_size = 0;
    size_t weights_const_size = 0;
    size_t required_mem_size = 0;
    for (auto node : _program->get_processing_order()) {
        if (node->is_type<fully_connected>() || node->is_type<convolution>() || node->is_type<deconvolution>())
            weights_const_size += get_buffer_size(*node);
        else if (node->is_type<data>())
            total_const_size += get_buffer_size(*node);
    }

    // Sum all weights constants for each stream
    required_mem_size += weights_const_size * _config.get_property(ov::streams::num);
    // Add all other constants (shared between streams)
    required_mem_size += total_const_size - weights_const_size;

    if (required_mem_size != 0) {
        const size_t required_weights_cache_capacity = 3;
        const size_t max_device_mem_size = _engine.get_device_info().max_global_mem_size;
        const size_t max_weights_cache_capacity = max_device_mem_size / required_mem_size;

        if (max_weights_cache_capacity > 1)
            _weights_cache_capacity = std::min(max_weights_cache_capacity, required_weights_cache_capacity);
    }
}

network::output_chains_map::iterator network::add_output_chain(std::shared_ptr<primitive_inst>& p_inst) {
    std::vector<primitive_inst*> chain;
    std::stack<const primitive_inst*> candidates;
    auto& eng = get_engine();
    const auto& mem_orig = p_inst->output_memory();

    auto add_mdata_chain = [&](primitive_inst* p_inst) {
        auto mdata_ptr = dynamic_cast<mutable_data_inst*>(p_inst);
        if (!mdata_ptr)
            return;
        // special handling for mutable data, which can share
        // its attached memory with both its inputs and outputs
        for (auto& dep : p_inst->dependencies()) {
            // check dependencies
            if (eng.is_the_same_buffer(mem_orig, dep.first->output_memory())) {
                chain.push_back(const_cast<primitive_inst*>(dep.first));
            }
            // then second order dependencies
            for (auto& second_dep : dep.first->dependencies()) {
                if (eng.is_the_same_buffer(mem_orig, second_dep.first->output_memory())) {
                    chain.push_back(const_cast<primitive_inst*>(second_dep.first));
                }
            }
        }

        //then users
        const auto& user_ids = mdata_ptr->get_user_ids();
        for (const auto& id : user_ids) {
            auto usr_prim = get_primitive(id).get();
            if (eng.is_the_same_buffer(mem_orig, usr_prim->output_memory())) {
                chain.push_back(usr_prim);
            }
        }
    };

    if (p_inst->can_be_optimized()) {
        candidates.push(p_inst.get());
    } else {
        chain.push_back(p_inst.get());
    }
    add_mdata_chain(p_inst.get());

    // find all dependencies that are 'optimized'
    while (!candidates.empty()) {
        auto cand = candidates.top();
        candidates.pop();
        const auto& mem_cand = cand->output_memory();
        if (eng.is_the_same_buffer(mem_orig, mem_cand)) {
            auto nc_cand = const_cast<primitive_inst*>(cand);
            chain.push_back(nc_cand);
            add_mdata_chain(nc_cand);
        }

        for (auto& dep : cand->dependencies()) {
            if (dep.first->can_be_optimized()) {
                candidates.push(dep.first);
            } else {
                const auto& mem_dep = dep.first->output_memory();
                if (eng.is_the_same_buffer(mem_orig, mem_dep)) {
                    auto nc_dep = const_cast<primitive_inst*>(dep.first);
                    chain.push_back(nc_dep);
                    add_mdata_chain(nc_dep);
                }
            }
        }
    }

    std::sort(chain.begin(), chain.end());
    chain.erase(std::unique(chain.begin(), chain.end()), chain.end());
    return _output_chains.insert({ p_inst->id(), chain }).first;
}

std::vector<event::ptr> network::set_output_memory(const primitive_id& id, memory::ptr mem_new) {
    GPU_DEBUG_TRACE_DETAIL << "Set output " << id << " " << mem_new->get_layout().to_short_string() << std::endl;
    std::shared_ptr<primitive_inst> p_inst;
    std::vector<event::ptr> ret_ev;
    p_inst = find_primitive(id);

    OPENVINO_ASSERT(p_inst != nullptr, "[GPU] topology doesn't contain primitive: ", id);

    auto iter = std::find(_outputs.begin(), _outputs.end(), p_inst);
    if (iter == _outputs.end())
        throw std::runtime_error("primitive: " + id + " is not a network output");

    auto& eng = get_engine();
    // locate primitive chain for this output
    // if no chain found - add it
    auto o_iter = _output_chains.find(id);
    if (o_iter == _output_chains.end()) {
        o_iter = add_output_chain(p_inst);
    }

    for (auto& prim : o_iter->second) {
        auto mem = mem_new;
        if (!prim->is_dynamic() && mem_new && prim->output_memory_ptr())
            mem = eng.reinterpret_buffer(*mem_new, prim->output_memory().get_layout());

        ret_ev.push_back(prim->set_output_memory(mem));
        if (!_reset_arguments &&
            (prim->type() != cldnn::data::type_id() && !(prim->type() == cldnn::mutable_data::type_id() && prim->dependencies().empty()))) {
            prim->set_arguments();
        }
    }
    return ret_ev;
}

void cldnn::network::check_names() {
    for (auto const& prim : _primitives) {
        if (find_in_internal_networks(prim.first) != nullptr)
            CLDNN_ERROR_MESSAGE("Network", "Found primitive with id: " + prim.first + "in anotother network.");
    }
}

std::shared_ptr<primitive_inst> cldnn::network::find_primitive(const primitive_id& id) const {
    std::shared_ptr<primitive_inst> ret;

    if (_primitives.find(id) != _primitives.end())
        return _primitives.at(id);

    return find_in_internal_networks(id);
}

std::shared_ptr<primitive_inst> cldnn::network::find_in_internal_networks(const primitive_id& id) const {
    std::shared_ptr<primitive_inst> ret;

    for (auto const& prim : _primitives) {
        if (prim.second->type() == condition::type_id()) {  // currently only condition inst contains mini networks
            auto cond_inst = std::static_pointer_cast<condition_inst>(prim.second);
            ret = cond_inst->get_net_true()->find_primitive(id);
            if (ret != nullptr)
                return ret;
            ret = cond_inst->get_net_false()->find_primitive(id);
            if (ret != nullptr)
                return ret;
        }
    }
    return nullptr;
}

std::string network::get_primitive_info(const primitive_id& id) const {
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

bool network::is_cpu_impl(const primitive_id& id) const {
    auto prim_inst = find_primitive(id);

    OPENVINO_ASSERT(prim_inst, "[GPU] Can't get implementation type, since topology ",
                               "doesn't contain primitive with requested id: ", id);

    return prim_inst->get_impl() ? prim_inst->get_impl()->is_cpu() : true;
}

std::string network::get_implementation_info(const primitive_id& id) const {
    return _program->get_implementation_info(id);
}

layout network::get_node_output_layout(const primitive_id& output_id) const {
    auto res = std::find_if(_outputs.begin(), _outputs.end(), [&](const std::shared_ptr<primitive_inst>& v) {
        return v->id() == output_id;
    });
    OPENVINO_ASSERT(res != _outputs.end(), "[GPU] Couldn't get output layout for ", output_id, ". Output with such name is not found in the outputs list");

    return (*res)->get_node_output_layout();
}

memory::ptr network::get_output_memory(const primitive_id& output_id) {
    return get_primitive(output_id)->output_memory_ptr();
}

layout network::get_output_layout(const primitive_id& output_id) const {
    return get_primitive(output_id)->get_output_layout();
}

void network::allocate_primitives() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("allocate_primitives");
    const auto& ao = _program->get_allocating_order();
    for (auto& node_id : ao) {
        allocate_primitive_instance(_program->get_node(node_id));
    }

    auto& po = _program->get_processing_order();

    for (auto const& node : po) {
        if (node->get_preferred_impl_type() == impl_types::onednn) {
            size_t eltw_dep = 0;
            for (auto& fused_op : node->get_fused_primitives()) {
                if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                    // If it is first sum, reuse the buffer
                    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(*node, fused_op);
                    if (fusing_type != add_fusing_type::sum || eltw_dep != 0)
                        continue;
                    if (!fused_op.has_outer_dep())
                        continue;
                    eltw_dep = fused_op.outer_dep_start_idx;
                    auto& eltw_in = node->get_dependency(eltw_dep);
                    if (_primitives.find(eltw_in.id()) != _primitives.end() && _primitives.find(node->id()) != _primitives.end()) {
                        auto& eltw_inst = _primitives.at(eltw_in.id());
                        auto& prim_inst = _primitives.at(node->id());
                        auto& eltw_mem = eltw_inst->output_memory();
                        auto new_mem = eltw_mem.get_engine()->reinterpret_buffer(eltw_mem, node->get_output_layout());
                        prim_inst->set_output_memory(new_mem);
                    }
                }
            }
        }
    }

    // Update the output memory address of optimized-out layer if it is not valid.
    for (auto const& node : po) {
        if (node->can_be_optimized() && !node->is_dynamic()) {
            auto opt_inst = _primitives.at(node->id());
            // build deps when prim_inst does not update dependencies yet.
            if (!node->get_dependencies().empty() && opt_inst->dependencies().empty()) {
                opt_inst->build_deps();
            }
            opt_inst->update_output_memory();
        }
    }

    // allocate intermediate buffers
    for (auto const& node : po) {
        auto prim = _primitives[node->id()];
        prim->allocate_internal_buffers();
    }
}

void network::configure_primitives_second_output() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("configure_primitives_second_output");
    std::map<cldnn::memory::ptr, std::vector<const cldnn::program_node*>> mutable_datas_ptrs;
    for (auto& inst : _primitives) {
        auto& node = inst.second->get_node();

        if (!node.is_type<mutable_data>())
            continue;

        mutable_datas_ptrs[node.as<mutable_data>().get_attached_memory_ptr()].push_back(&node);
    }

    for (auto item : mutable_datas_ptrs) {
        if (item.second.size() != 2)
            continue;

        auto is_first_node_input_md = [&](const cldnn::program_node* first,
                                          const cldnn::program_node* second) {
            for (auto user : first->get_users()) {
                for (auto next_user : user->get_users()) {
                    if (next_user == second)
                        return true;
                }
            }
            return false;
        };

        auto is_first_node_input = is_first_node_input_md(item.second[0], item.second[1]);

        auto input_md_inst = is_first_node_input ? _primitives[item.second[0]->id()] : _primitives[item.second[1]->id()];
        auto output_md_inst = is_first_node_input ? _primitives[item.second[1]->id()] : _primitives[item.second[0]->id()];

        output_md_inst->set_output_memory(input_md_inst->output_memory_ptr(), false);
    }
}

void network::build_insts_deps() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("build_insts_deps");
    for (auto& inst : _primitives) {
        inst.second->build_deps();
        inst.second->configure_shape_of_dependencies();
    }
}

void network::build_exec_order() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("build_exec_order");
    if (!_is_dynamic) {
        for (auto& node : _program->get_processing_order()) {
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                add_to_exec_order(node->id());
            }
        }
    } else {
        auto is_runtime_optimized_concat = [&](const program_node* node) {
            return (node->is_dynamic() && node->is_type<concatenation>() && node->can_be_optimized());
        };
        auto is_allowed_pred_for_runtime_optimized_concat = [&](const program_node* node) {
            return (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty()) &&
                    node->get_users().size() == 1 && is_runtime_optimized_concat(node->get_users().front()));
        };
        for (auto& node : _program->get_processing_order()) {
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                if (is_allowed_pred_for_runtime_optimized_concat(node)) {
                    continue;
                } else if (is_runtime_optimized_concat(node)) {
                    // For in-place concat applied at runtime, we need to do update_shape for all other predecessors of the concat user.
                    // i.e., We need to make sure that all the preds of them are already updated too.
                    for (auto dep : node->get_dependencies()) {
                        if (!dep.first->is_type<data>()) {
                            add_to_exec_order(dep.first->id());
                        }
                    }
                }
                add_to_exec_order(node->id());
            }
        }
    }
}
void network::add_to_exec_order(const primitive_id& id) {
    auto inst = get_primitive(id);
    _exec_order.push_back(inst);
}

std::map<primitive_id, network_output> network::execute(const std::vector<event::ptr>& dependencies) {
    execute_impl(dependencies);

    auto output_ids = get_output_ids();
    std::map<primitive_id, network_output> result;
    for (auto& id : output_ids) {
        result.emplace(id, get_output(id));
    }
    return result;
}


void network::execute_impl(const std::vector<event::ptr>& events) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "NetworkImpl::Execute");
    int64_t curr_iter = -1;
    GPU_DEBUG_GET_INSTANCE(debug_config);
#ifdef GPU_DEBUG_CONFIG
    curr_iter = iteration++;
#endif

    // Wait for previous execution completion
    reset_execution(false);
    GPU_DEBUG_IF(debug_config->dump_runtime_memory_pool > 0) {
        GPU_DEBUG_COUT << "============================================================================" << std::endl;
        GPU_DEBUG_COUT << "Start network execution (net_id : " << get_id() << ", iter :" << curr_iter << ")" << std::endl;
    } else {
        GPU_DEBUG_TRACE << "============================================================================" << std::endl;
        GPU_DEBUG_TRACE << "Start network execution (net_id : " << get_id() << ", iter :" << curr_iter << ")" << std::endl;
    }

    std::vector<memory::ptr> in_out_mem;
    auto is_surface_lock_check_needed = [&](const shared_mem_type& shared_mem_type) {
        return shared_mem_type == shared_mem_type::shared_mem_vasurface ||
               shared_mem_type == shared_mem_type::shared_mem_dxbuffer ||
               shared_mem_type == shared_mem_type::shared_mem_image;
    };

    bool shared_mem_found = std::any_of(_in_out_shared_mem_types.begin(),
                                        _in_out_shared_mem_types.end(),
                                        is_surface_lock_check_needed);

    if (shared_mem_found) {
        for (auto& inst : _inputs) {
            if (inst->output_memory_ptr() &&
                is_surface_lock_check_needed(inst->output_memory_ptr()->get_internal_params().mem_type))
                in_out_mem.push_back(inst->output_memory_ptr());
        }

        for (auto& inst : _outputs) {
            if (inst->output_memory_ptr() &&
                is_surface_lock_check_needed(inst->output_memory_ptr()->get_internal_params().mem_type))
                in_out_mem.push_back(inst->output_memory_ptr());
        }
    }

    // We shouldn't call surfaces_lock::create() function constantly here, but due to
    // some changes in assembler code, performance drops in case if we move it under
    // `shared_mem_found` condition (it somehow connected with get_cl_queue() - this function call
    // makes asm faster for some reasons). So, as WA we keep this surfaces_lock::create() here
    // with empty memory vector and do nothing inside this function for saving performance
    // in some cases.
    auto surf_lock = surfaces_lock::create(get_engine().type(), in_out_mem, get_stream());

    set_arguments();
    GPU_DEBUG_IF(debug_config->list_layers == 1) {
        for (auto& inst : _exec_order) {
            GPU_DEBUG_COUT << inst->id() << std::endl;
            if (inst->get_node().is_type<loop>()) {
                auto& loop_node = inst->get_node().as<loop>();
                for (auto& prim : loop_node.get_body_program()->get_processing_order()) {
                    GPU_DEBUG_COUT << "\t" << prim->id() << std::endl;
                }
            } else if (inst->get_node().is_type<condition>()) {
                auto& cond_node = inst->get_node().as<condition>();
                GPU_DEBUG_COUT << "* Branch_True" << std::endl;
                for (auto& prim : cond_node.get_branch_true().inner_program->get_processing_order()) {
                    GPU_DEBUG_COUT << "\t" << prim->id() << std::endl;
                }
                GPU_DEBUG_COUT << "* Branch_False" << std::endl;
                for (auto& prim : cond_node.get_branch_false().inner_program->get_processing_order()) {
                    GPU_DEBUG_COUT << "\t" << prim->id() << std::endl;
                }
            }
        }
        if (!is_internal()) exit(0);
    }
    auto get_iteration_prefix = [](int64_t iter) {
        if (iter < 0)
            return std::string("");
        return std::to_string(iter) + "_";
    };

    // This extra flush command is needed for dynamic models in case of out_of_order operating mode since
    // it reduces `bubbles` number in pipeline and GPU's idle time by timely flushing new kernels to device.
    // The freqency of flushing (16) is selected empirically, see details in tickets 116365, 116287.
    const bool is_out_of_order_queue = get_stream().get_queue_type() == QueueTypes::out_of_order;
    const bool needs_flushing = _is_dynamic && is_out_of_order_queue;
    const size_t flush_frequency = needs_flushing ? 16 : 0;
    size_t executed_prims = 0;

    for (auto& inst : _exec_order) {
        // Load binary dump for input layers
        GPU_DEBUG_IF(!debug_config->load_layers_raw_dump.empty()) {
            const std::string layer_name = inst->id();
            auto files = debug_config->get_filenames_for_matched_layer_loading_binaries(layer_name);
            if (!files.empty()) {
                if (inst->is_input()) {
                    // Loading binary dumps for output tensors of input-layers : only one output exists or index(dstN) exists
                    auto dump_file = debug_config->get_matched_from_filelist(files, "_dst0__");
                    OPENVINO_ASSERT((files.size() == 1 || dump_file.length() != 0), "Unexpected binary dump for input layer");

                    OPENVINO_ASSERT(files.size() == get_primitive(inst->id())->outputs_memory_count(), "Mis-match dump file count");

                    for (size_t i = 0; i < get_primitive(inst->id())->outputs_memory_count(); i++) {
                        auto dump_file = files[0];
                        if (files.size() > 1 || get_primitive(inst->id())->outputs_memory_count() != 1) {
                            std::string pattern = "_dst" + std::to_string(i) + "__";
                            dump_file = debug_config->get_matched_from_filelist(files, pattern);
                        }
                        OPENVINO_ASSERT((dump_file.length() > 0), "Could not find expected pattern '_dst[N]__' for binary dump");
                        GPU_DEBUG_COUT  << " Load binary dump : " << dump_file << " for " << layer_name << std::endl;

                        std::vector<uint8_t> bin = ov::util::load_binary(dump_file);
                        OPENVINO_ASSERT(!bin.empty(), "Failure loading binary from OV_GPU_LoadDumpRawBinary : " + dump_file);

                        auto output_mem = get_primitive(layer_name)->output_memory_ptr(i);
                        OPENVINO_ASSERT(output_mem->size() == bin.size(), "memory size mis-match for OV_GPU_LoadDumpRawBinary : " + layer_name
                                        + "\n Expected size : " + to_string(output_mem->size()) + ", Binary : " + to_string(bin.size()));

                        output_mem->copy_from(get_stream(), static_cast<void *>(&bin[0]), true);
                    }
                } else {
                    auto check_dst = debug_config->get_matched_from_filelist(files, "_dst0__");
                    OPENVINO_ASSERT(check_dst.length() == 0, "Expected to load binaries for inputs of " + layer_name);

                    // Loading input tensors for any layer
                    auto dump_file = debug_config->get_matched_from_filelist(files, "_src0__");
                    OPENVINO_ASSERT(dump_file.length() != 0, "Could not find expected pattern '_src[N]__' for binary dump input : " + layer_name);

                    OPENVINO_ASSERT(files.size() == get_primitive(inst->id())->dependencies().size(), "Mis-match dump file count");

                    for (size_t i = 0; i < get_primitive(inst->id())->dependencies().size(); i++) {
                        auto dump_file = files[0];
                        if (files.size() > 1 || get_primitive(inst->id())->dependencies().size() != 1) {
                            std::string pattern = "_src" + std::to_string(i) + "__";
                            dump_file = debug_config->get_matched_from_filelist(files, pattern);
                        }
                        OPENVINO_ASSERT((dump_file.length() > 0), "Could not find expected pattern '_src[N]__' for binary dump input");
                        GPU_DEBUG_COUT  << " Load binary dump : " << dump_file << " for input of " << layer_name << std::endl;

                        std::vector<uint8_t> bin = ov::util::load_binary(dump_file);
                        OPENVINO_ASSERT(!bin.empty(), "Failure loading binary from OV_GPU_LoadDumpRawBinary : " + dump_file);

                        auto input_mem = get_primitive(inst->id())->dep_memory_ptr(i);
                        OPENVINO_ASSERT(input_mem->size() == bin.size(), "memory size mis-match for OV_GPU_LoadDumpRawBinary : " + layer_name);

                        input_mem->copy_from(get_stream(), static_cast<void *>(&bin[0]), true);
                    }
                }
            }
        }

        // Dump input buffers of 'inst'
        GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
            const std::string layer_name = inst->id();
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                std::cerr << inst->id() << std::endl;
            }

            GPU_DEBUG_IF(debug_config->is_target_iteration(curr_iter) &&
                        debug_config->dump_layers_dst_only == 0 && debug_config->is_layer_for_dumping(layer_name)) {
                std::string debug_str_for_bin_load = " Command for loading : OV_GPU_LoadDumpRawBinary=\"" + layer_name + ":";
                for (size_t i = 0; i < get_primitive(inst->id())->dependencies().size(); i++) {
                    std::string name = "program" + std::to_string((get_program() != nullptr) ? get_program()->get_id() : 0) +
                                        "_network" + std::to_string(get_id()) +
                                        "_" + get_iteration_prefix(curr_iter) +
                                        layer_name + "_src" + std::to_string(i);
                    auto input_mem = get_primitive(inst->id())->dep_memory_ptr(i);
                    auto dep = inst->dependencies().at(i);
                    auto input_layout = dep.first->get_output_layout(dep.second);
                    GPU_DEBUG_IF(debug_config->dump_layers_binary) {
                        // Binary dump : raw
                        auto filename = get_file_path_for_binary_dump(input_layout, name);

                        mem_lock<char, mem_lock_type::read> lock(input_mem, get_stream());
                        ov::util::save_binary(filename, lock.data(), input_mem->size());
                        GPU_DEBUG_COUT  << " Dump layer src : " << layer_name << " to " << filename << std::endl;
                        debug_str_for_bin_load += (filename + ",");
                    } else {
                        log_memory_to_file(input_mem,
                                        input_layout,
                                        get_stream(),
                                        name,
                                        debug_config->dump_layers_raw);
                    }
                }

                GPU_DEBUG_IF(debug_config->dump_layers_binary && !inst->is_input()) {
                    debug_str_for_bin_load[debug_str_for_bin_load.size()-1] = '\"';
                    GPU_DEBUG_COUT << debug_str_for_bin_load << std::endl;;
                }
            }
        }

        execute_primitive(inst, events);
        executed_prims++;
        if (needs_flushing && executed_prims % flush_frequency == 0)
            get_stream().flush();

        // Dump output buffers of 'inst'
        GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
            get_stream().finish();
            const std::string layer_name = inst->id();
            auto prog_id = ((get_program() != nullptr) ? get_program()->get_id() : 0);
            auto net_id = get_id();
            GPU_DEBUG_IF(debug_config->is_target_iteration(curr_iter) &&
                        debug_config->is_layer_for_dumping(layer_name, inst->is_output(), inst->is_input())) {
                std::string debug_str_for_bin_load = " Command for loading : OV_GPU_LoadDumpRawBinary=\""
                                                        + layer_name + ":";
                for (size_t i = 0; i < get_primitive(layer_name)->outputs_memory_count(); i++) {
                    std::string name = "program" + std::to_string(prog_id) +
                                        "_network" + std::to_string(net_id) +
                                        "_" + get_iteration_prefix(curr_iter) +
                                        layer_name + "_dst" + std::to_string(i);
                    auto output_mem = get_primitive(layer_name)->output_memory_ptr(i);
                    GPU_DEBUG_IF(debug_config->dump_layers_binary) {
                        // Binary dump : raw
                        auto output_layout = inst->get_output_layout(i);
                        auto filename = get_file_path_for_binary_dump(output_layout, name);

                        mem_lock<char, mem_lock_type::read> lock(output_mem, get_stream());
                        ov::util::save_binary(filename, lock.data(), output_mem->size());
                        GPU_DEBUG_COUT  << " Dump layer dst : " << layer_name << " to " << filename << std::endl;
                        debug_str_for_bin_load += (filename + ",");
                    } else {
                        // Text dump
                        log_memory_to_file(output_mem, inst->get_output_layout(i), get_stream(), name, debug_config->dump_layers_raw);
                    }
                }

                GPU_DEBUG_IF(debug_config->dump_layers_binary && inst->is_input()) {
                    debug_str_for_bin_load[debug_str_for_bin_load.size()-1] = '\"';
                    GPU_DEBUG_COUT << debug_str_for_bin_load << std::endl;;
                }
            }
        }
    }

    // print '-data_shape' option for benchmark_app
    GPU_DEBUG_IF(debug_config->print_input_data_shapes == 1) {
        std::stringstream data_shape_str;
        auto add_string = [&data_shape_str](std::string str) {
            data_shape_str << ((data_shape_str.rdbuf()->in_avail() == 0) ? " -data_shape " : ",") << str;
        };

        for (auto& inst : _exec_order) {
            auto name = inst->id();
            auto pos = name.find(':');
            auto type = name.substr(0, pos);
            name.erase(0, pos + 1);
            if (inst->is_input() && type == "parameter") {
                add_string(name + inst->get_output_layout().get_partial_shape().to_string());
            }
        }

        GPU_DEBUG_COUT << "[program:" << std::setw(2) << ((get_program() != nullptr) ? get_program()->get_id() : 0)
                       << "|network:" << std::setw(2) << get_id() << "|iter:" << std::setw(4) << curr_iter <<  "] benchmark_app cmd: "
                       << data_shape_str.str() << std::endl;
    }

    GPU_DEBUG_IF(!debug_config->dump_graphs.empty() && debug_config->is_target_iteration(curr_iter)) {
        auto get_fixed_str = [](int value, int length = 2) -> std::string {
            std::ostringstream ss;
            ss << std::setw(length) << std::setfill('0') << std::to_string(value);
            return ss.str();
        };
        std::string path = get_dir_path(get_config());
        if (!path.empty()) {
            std::ofstream ofs(path + "cldnn_program_exec_p" + get_fixed_str(get_program()->get_id()) + "_n" + get_fixed_str(get_id())
                              + "_" + get_fixed_str(curr_iter, 5) + ".graph");
            dump_graph_init(ofs, *get_program(), [&](const primitive_id& id) -> std::shared_ptr<primitive_inst> {
                return get_primitive(id);
            });
        }
    }

    // Store events only in case of OOO queue or enabled Profiling
    auto store_events = is_out_of_order_queue || _enable_profiling;
    if (store_events) {
        if (_program != nullptr) {
        for (auto& inst : _program->get_processing_order()) {
            // Special handling for mutable data. The event should be the same as the user or dependency with highest
            // processing_num as the mutable_data can be updated when is both user or dependency.
            if (inst->is_type<mutable_data>()) {
                decltype(_program->get_processing_order().get_processing_number(inst)) proc_num = 0;
                for (auto& user : inst->get_users()) {
                    auto user_proc_num = _program->get_processing_order().get_processing_number(user);
                    if (user_proc_num > proc_num) {
                        _events[inst->id()] = _events[user->id()];
                        proc_num = user_proc_num;
                    }
                }

                if (!inst->get_dependencies().empty()) {
                    for (auto& dep : inst->get_dependencies()) {
                        auto dep_proc_num = _program->get_processing_order().get_processing_number(dep.first);
                        if (dep_proc_num > proc_num) {
                            _events[inst->id()] = _events[dep.first->id()];
                            proc_num = dep_proc_num;
                        }
                    }
                }
            }
        }
        }

        for (auto& dout : _data_outputs) {  // data primitives are not executed so if they are marked as output we need to add
                                            // them valid events manually
            _events[dout->id()] = get_stream().create_user_event(true);
        }
    }

    for (auto& prim : _primitives) {
        prim.second->reset_output_change();
    }

    // Using output of previous network as input to another one may cause hazard (in OOOQ mode) if user would not
    // provide proper event to execution. Flushing pipeline should prevent this kind of issues.
    // In scenarios with a big number of very small networks it can provide performance drop.
    get_stream().flush();

    // Deallocate events from the previos iteration
    _old_events.clear();

    GPU_DEBUG_IF(debug_config->dump_runtime_memory_pool > 0) {
        get_memory_pool().dump(get_id());
    }
}

std::vector<primitive_id> network::get_input_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_inputs.size());
    for (auto const& input : _inputs) ret.push_back(input->id());
    return ret;
}

std::vector<layout> network::get_input_layouts() const {
    std::vector<layout> ret;
    ret.reserve(_inputs.size());
    for (auto const& input : _inputs) ret.push_back(input->output_memory_ptr()->get_layout());
    return ret;
}

std::vector<primitive_id> network::get_output_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_outputs.size());
    for (auto const& output : _outputs) ret.push_back(output->id());
    return ret;
}

std::vector<primitive_id> network::get_executed_primitive_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_exec_order.size());
    for (auto const& executed_primitive : _exec_order) {
        ret.push_back(executed_primitive->id());
    }
    return ret;
}

std::vector<primitive_id> network::get_all_primitive_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        if (primitive.second->can_be_optimized())
            ret.push_back("_optimized_");
        else
            ret.push_back(primitive.second->id());
    return ret;
}

std::vector<primitive_id> network::get_all_primitive_org_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives) ret.push_back(primitive.second->org_id());
    return ret;
}

const program::primitives_info& network::get_primitives_info() const {
    return (_program == nullptr) ? _prims_info : _program->get_primitives_info();
}

const program::graph_optimizer_info& network::get_optimizer_passes_info() const {
    return _program->get_optimizer_passes_info();
}

std::map<primitive_id, primitive_id> network::get_ext_id_mapping() const {
    if (_program == nullptr) {
        return _ext_id_mapping;
    }

    std::map<primitive_id, primitive_id> result;
    for (auto& prim : _primitives) {
        result.emplace(prim.first, prim.second->get_node().get_primitive()->origin_op_name);
    }
    for (auto& opt_id : _program->get_optimized_out()) {
        std::string ext_id = opt_id;
        if (opt_id.find(":") != std::string::npos) {
            ext_id = opt_id.substr(opt_id.find(":") + 1, opt_id.length());
        }
        result.emplace(opt_id, ext_id);
    }
    return result;
}

std::shared_ptr<primitive_inst> network::get_primitive(const primitive_id& id) {
    if (!_primitives.count(id))
        allocate_primitive_instance(_program->get_node(id));

    return _primitives.at(id);
}

std::shared_ptr<const primitive_inst> network::get_primitive(const primitive_id& id) const {
    OPENVINO_ASSERT(_primitives.count(id) == 1, "[GPU] Can't get primitive with ", id, " id: primitive with such name hasn't been found in processing order");
    return _primitives.at(id);
}

std::vector<primitive_inst*> network::get_primitives(const std::vector<primitive_id>& ids) {
    std::vector<primitive_inst*> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) {
        return get_primitive(id).get();
    });
    return result;
}

std::vector<std::pair<primitive_inst*, int>> network::get_primitives(const std::vector<std::pair<program_node*, int>>& nodes) {
    std::vector<std::pair<primitive_inst*, int>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const std::pair<program_node*, int>& node) {
        return std::make_pair(get_primitive(node.first->id()).get(), node.second);
    });
    return result;
}

void network::execute_primitive(const std::shared_ptr<primitive_inst>& primitive,
                                const std::vector<event::ptr>& events) {
    event::ptr ev = primitive->execute(events);

    // Collect events under any of the following conditions:
    // 1) OOO queue execution
    // 2) Profiling mode is enabled
    // 3) Primitive has CPU user or primitive is output
    if (get_stream().get_queue_type() == QueueTypes::out_of_order || _enable_profiling || primitive->needs_completion_event()) {
        auto id = primitive->id();
        _events.insert({id, ev});
    }
}

void network::allocate_primitive_instance(program_node const& node) {
    if (_primitives.count(node.id()))
        return;

    GPU_DEBUG_TRACE_DETAIL << node.id() << ": allocate primitive instance" << std::endl;

    auto inst = node.type()->create_instance(*this, node);

    std::function<bool(const program_node&)> is_mutable_input = [&is_mutable_input](const program_node& node) {
        for (auto& dep : node.get_dependencies()) {
            const auto dep_node = dep.first;
            if (dep_node->is_type<input_layout>() || dep_node->is_type<mutable_data>() || (dep_node->is_type<read_value>() && !dep_node->can_be_optimized())) {
                return true;
            }
            if (dep_node->can_be_optimized()) {
                if (is_mutable_input(*dep_node) || dep_node->is_dynamic()) {
                    return true;
                }
            }
        }
        return false;
    };

    if (is_mutable_input(node)) {
        inst->set_mutable_input(true);
    }

    if (inst->is_dynamic()) {
        _is_dynamic = true;
    }

    _primitives[node.id()] = inst;
    if (node.is_type<input_layout>()) {
        if (inst->output_memory_ptr())
            _in_out_shared_mem_types.push_back(inst->output_memory_ptr()->get_internal_params().mem_type);
        _inputs.push_back(inst);
    }
    if (node.is_output()) {
        if (inst->output_memory_ptr())
            _in_out_shared_mem_types.push_back(inst->output_memory_ptr()->get_internal_params().mem_type);
        _outputs.push_back(inst);
        if (node.is_type<data>())
            _data_outputs.push_back(inst);
    }
    if (node.is_type<kv_cache>()) {
       kv_cache_ids.push_back(node.id());
    }
    if (auto state_prim = std::dynamic_pointer_cast<memory_state::variable>(inst)) {
        auto prim = inst->get_node().get_primitive();
        set_variables_state_info(state_prim->variable_id(), node.get_output_layout(0), state_prim->get_user_specified_type(), prim.get());
    }
    if (node.is_constant())
        transfer_memory_to_device(inst, node);
}

void network::transfer_memory_to_device(std::shared_ptr<primitive_inst> instance, program_node const& node) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "NetworkImpl::TransferMemory");
    auto& inst_mem = instance->output_memory();
    auto alloc_type = inst_mem.get_allocation_type();

    auto users = node.get_users();
    if (users.size() == 1
        && users.front()->is_type<reshape>()
        && users.front()->is_dynamic())
            return;

    // Do not transfer memory if a user requires lockable memory.
    // If memory is used in both gpu and cpu implementations, primitive itself is responsible for correct allocation type
    if (node.need_lockable_memory())
        return;

    if (!get_engine().supports_allocation(allocation_type::usm_device))
        return;

    if (get_engine().get_device_info().dev_type != device_type::discrete_gpu)
        return;

    if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
        // Allocate and transfer memory
        auto device_mem = inst_mem.get_engine()->allocate_memory(inst_mem.get_layout(), allocation_type::usm_device, false);
        device_mem->copy_from(get_stream(), inst_mem);
        GPU_DEBUG_LOG << "[" << node.id() << ": constant]" << std::endl;
        _memory_pool->release_memory(&inst_mem, node.id(), get_id());
        instance->set_output_memory(device_mem);
    }
}

void network::set_variable(const std::string& name, const std::shared_ptr<ov::intel_gpu::VariableStateBase>& variable) {
    GPU_DEBUG_TRACE_DETAIL << "Set variable " << name << " " << variable->get_layout().to_short_string() << std::endl;
    _variables_states[name] = variable;
}

bool network::has_variable(const std::string &variable_id) const {
    return _variables_states.find(variable_id) != _variables_states.end();
}

ov::intel_gpu::VariableStateBase& network::get_variable(const std::string &variable_id) const {
    auto it = _variables_states.find(variable_id);
    OPENVINO_ASSERT(it != _variables_states.end(), "[GPU] ", variable_id, " variable not found");
    return *it->second;
}

const ov::intel_gpu::VariableStateInfo& network::get_variable_info(const std::string &variable_id) const {
    auto it = _variables_state_info.find(variable_id);
    OPENVINO_ASSERT(it != _variables_state_info.end(), "[GPU] ", variable_id, " variable info not found");
    return it->second;
}

const ov::intel_gpu::VariablesMap& network::get_variables() const {
    return _variables_states;
}

const ov::intel_gpu::VariablesInfoMap& network::get_variables_info() const {
    return _variables_state_info;
}

void network::set_variables_state_info(const std::string& variable_id,
                                       const layout& variable_layout,
                                       ov::element::Type user_specified_type,
                                       const primitive* p) {
    _variables_state_info.emplace(variable_id, ov::intel_gpu::VariableStateInfo{variable_id, variable_layout, user_specified_type});

    _variables_state_info.at(variable_id).m_primitives.insert(p);
}


}  // namespace cldnn
