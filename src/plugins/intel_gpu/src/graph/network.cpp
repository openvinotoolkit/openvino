// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/read_value.hpp"

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
#include "paged_attention_inst.h"
#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "mutable_data_inst.h"
#include "condition_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"
#include "kv_cache_inst.h"
#include "program_helpers.h"
#include "program_dump_graph.h"

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

#include "debug_helper.hpp"
#ifdef GPU_DEBUG_CONFIG
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
void wait_for_the_turn() {}
#endif
}  // namespace

static uint32_t get_unique_net_id() {
    static std::atomic<uint32_t> id_gen{0};
    return ++id_gen;
}

/*
Network will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by propagate_constants
opt pass).
*/
network::network(program::ptr program, stream::ptr stream, bool is_internal, bool is_primary_stream)
    : _program(program)
    , _config(program->get_config())
    , _engine(program->get_engine())
    , _stream(stream)
    , _memory_pool(new memory_pool(program->get_engine()))
    , _internal(is_internal)
    , _is_primary_stream(is_primary_stream)
    , _enable_profiling(program->get_config().get_property(ov::enable_profiling))
    , _reset_arguments(true)
    , _shape_predictor(new ShapePredictor(&program->get_engine(), program->get_config().get_property(ov::intel_gpu::buffers_preallocation_ratio))) {
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
    build_insts_deps();
    build_exec_order();
    validate_primitives();
    add_default_output_chains();
}

network::network(program::ptr program, bool is_internal, bool is_primary_stream)
    :  network(program, program->get_engine().create_stream(program->get_config()), is_internal, is_primary_stream) {}

network::network(engine& engine,
                 const topology& topo,
                 const ExecutionConfig& config,
                 bool is_internal,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor)
    : network(program::build_program(engine, topo, config, task_executor, is_internal), is_internal, true) {}

network::network(engine& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const ExecutionConfig& config,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                 bool is_internal)
    : network(program::build_program(engine, nodes, config, task_executor, is_internal), is_internal, true) {}

network::network(program::ptr program, uint16_t stream_id)
    : network(program, program->get_engine().create_stream(program->get_config()), false, stream_id == 0) {}

network::network(program::ptr program, stream::ptr stream, uint16_t stream_id)
    : network(program, stream, false, stream_id == 0) {}

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
    return std::make_shared<network>(program, stream, is_internal, is_primary_stream);
}

network::ptr network::allocate_network(engine& engine, program::ptr program, bool is_internal, bool is_primary_stream) {
    auto stream = engine.create_stream(program->get_config());
    return std::make_shared<network>(program, stream, is_internal, is_primary_stream);
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

                // (dynamic) -> static optimizable reshape -> static optimizable reshape -> some_op
                // In that case, it is a limit about second reshape.
                auto prim = dep.first->get_impl_params()->desc;
                if (dep.first->can_be_optimized() && (dep.first->is_dynamic() ||
                                                      dep.first->output_memory_ptr() == nullptr ||
                                                      prim->type == read_value::type_id()))
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
    auto primitive_inst = find_primitive(id);

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
        // Add cand inst to the chain when cand's output is not allocated yet.
        if (!p_inst->outputs_allocated()
            || eng.is_the_same_buffer(mem_orig, mem_cand)) {
            auto nc_cand = const_cast<primitive_inst*>(cand);
            chain.push_back(nc_cand);
            add_mdata_chain(nc_cand);
        }

        for (auto& dep : cand->dependencies()) {
            if (dep.first->can_be_optimized()) {
                candidates.push(dep.first);
            } else {
                if (dep.first->outputs_allocated()) {
                    const auto& mem_dep = dep.first->output_memory();
                    // Add dep inst to the chain when dep's output is not allocated yet.
                    if (!p_inst->outputs_allocated()
                        || eng.is_the_same_buffer(mem_orig, mem_dep)) {
                        auto nc_dep = const_cast<primitive_inst*>(dep.first);
                        chain.push_back(nc_dep);
                        add_mdata_chain(nc_dep);
                    }
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
    std::vector<event::ptr> ret_ev;
    std::shared_ptr<primitive_inst> p_inst = find_primitive(id);

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

std::shared_ptr<primitive_inst> cldnn::network::find_primitive(const primitive_id& id) const {
    auto it = _primitives.find(id);
    OPENVINO_ASSERT(it != _primitives.end(), "[GPU] Network doesn't contain primitive ", id);
    return it->second;
}

std::string network::get_primitive_info(const primitive_id& id) const {
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

bool network::does_node_need_lockable_output(const primitive_id& id) const {
    auto prim_inst = find_primitive(id);

    const auto& node = prim_inst->get_node();
    if (node.is_type<input_layout>()) {
        for (const auto& user : node.get_users()) {
            const auto& lockable_input_ids = user->get_lockable_input_ids();
            if (lockable_input_ids.count(user->get_dependency_index(node))) {
                return true;
            }
        }

        return false;
    } else {
        return prim_inst->get_impl() ? prim_inst->get_impl()->is_cpu() : true;
    }
}

std::string network::get_implementation_info(const primitive_id& id) const {
    return _program->get_implementation_info(id);
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
        inst.second->init_users();
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
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "NetworkImpl::Execute");
    NETWORK_DEBUG(*this);

    // Wait for previous execution completion
    reset_execution(false);

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

    execute_impl(dependencies);

    std::map<primitive_id, network_output> result;
    for (auto& inst : _outputs) {
        event::ptr ev = nullptr;
        const auto& id = inst->id();
        if (get_stream().get_queue_type() == QueueTypes::out_of_order || _enable_profiling)
            ev = _events.at(id);

        result.emplace(id, network_output(ev, inst->output_memory_ptr(0), get_stream_ptr(), inst->get_output_layout(0)));
    }
    return result;
}

void network::execute_impl(const std::vector<event::ptr>& events) {
    set_arguments();

    // This extra flush command is needed for dynamic models in both cases of out_of_order / in_order operating mode
    // since it reduces `bubbles` number in pipeline and GPU's idle time by timely flushing new kernels to device.
    // The freqency of flushing (16) is selected empirically, see details in tickets 116365, 116287, 139931.
    const bool is_out_of_order_queue = get_stream().get_queue_type() == QueueTypes::out_of_order;
    const bool needs_flushing = _is_dynamic;
    const size_t flush_frequency = needs_flushing ? 16 : 0;
    size_t executed_prims = 0;

    for (auto& inst : _exec_order) {
        NODE_DEBUG(*inst);

        execute_primitive(inst, events);
        executed_prims++;
        if (needs_flushing && executed_prims % flush_frequency == 0)
            get_stream().flush();
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
                if (is_mutable_input(*dep_node)) {
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

    if (!get_engine().get_device_info().has_separate_cache)
        return;

    if (node.is_shape_infer_dep())
        return;

    if (inst_mem.count() == 0)
        return;

    if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
        // Allocate and transfer memory
        auto device_mem = inst_mem.get_engine()->allocate_memory(inst_mem.get_layout(), allocation_type::usm_device, false);
        device_mem->copy_from(get_stream(), inst_mem);
        GPU_DEBUG_LOG << "[" << node.id() << ": constant]" << std::endl;
        _memory_pool->release_memory(&inst_mem, node.get_unique_id(), node.id(), get_id());
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
