// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/half.hpp"

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "assign_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"

#include "to_string_utils.h"
#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "mutable_data_inst.h"
#include "condition_inst.h"
#include "loop_inst.h"
#include "kernel_selector_helper.h"
#include "program_helpers.h"
#include "intel_gpu/runtime/itt.hpp"
#include "kernels_cache.hpp"
#include "compilation_context.hpp"

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

    const std::string perf_raw_csv_header = "prim_id,prim_type,stage,net_in_shapes,in_shapes,out_shapes,impl,iters,time_usec\n";
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
                auto& num_iters = std::get<1>(entry);
                int64_t time_avg = time / num_iters;
                std::string net_in_l_str = layouts_to_str(key.network_input_layouts);
                std::string in_l_str = layouts_to_str(key.input_layouts);
                std::string out_l_str = layouts_to_str(key.output_layouts);
                of << prim_id << ","
                << inst->desc()->type_string() << ","
                << key.stage << (key.cache_hit ? " (cache_hit)" : "") << ","
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

float convert_element(int32_t i) { return static_cast<float>(i); }

float convert_element(float f) { return f; }

float convert_element(half_t h) { return half_to_float(h); }

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
void dump(memory::ptr mem, stream& stream, std::ofstream& file_stream) {
    auto&& size = mem->get_layout().get_tensor();

    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto batch_size = std::max(std::min(debug_config->dump_layers_limit_batch, size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count() << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")" << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count() << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
            << ", original shape: " << size.to_string() << ")" << std::endl;
    }

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());
    std::stringstream buffer;

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
    file_stream << buffer.str();
}

template <>
void dump<uint32_t>(memory::ptr mem, stream& stream, std::ofstream& file_stream) {
    auto&& l = mem->get_layout();

    file_stream << "shape: ";
    file_stream << l.batch() << " ";
    file_stream << l.feature() << " ";
    file_stream << l.spatial(1) << " ";
    file_stream << l.spatial(0) << " ";
    file_stream << "(" << l.batch() * l.feature() * l.spatial(1) * l.spatial(0) << ")" << std::endl;

    mem_lock<uint32_t, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();

    for (cldnn::tensor::value_type b = 0; b < l.batch(); ++b) {
        for (cldnn::tensor::value_type f = 0; f < (cldnn::tensor::value_type)ceil_div(l.feature(), 32); ++f) {
            for (cldnn::tensor::value_type z = 0; z < l.spatial(2); ++z) {
                for (cldnn::tensor::value_type y = 0; y < l.spatial(1); ++y) {
                    for (cldnn::tensor::value_type x = 0; x < l.spatial(0); ++x) {
                        cldnn::tensor t(cldnn::batch(b), cldnn::feature(f), cldnn::spatial(x, y, z, 0));
                        size_t input_it = mem->get_layout().get_linear_offset(t);
                        file_stream << mem_ptr[input_it] << std::endl;
                    }
                }
            }
        }
    }
}

void log_memory_to_file(memory::ptr mem, stream& stream, std::string layerName) {
    std::cout << "Dump " << layerName << std::endl;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    std::string filename = layerName;
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
    filename = debug_config->dump_layers_path + filename + ".txt";

    std::ofstream file_stream(filename);
    auto mem_dt = mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        dump<float>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::f16)
        dump<half_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::bin)
        dump<uint32_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::i32)
        dump<int32_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::i8)
        dump<int8_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(mem, stream, file_stream);
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
void log_memory_to_file(memory::ptr, stream&, std::string) {}
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
network::network(program::ptr program, const ExecutionConfig& config, stream::ptr stream, bool is_internal, bool is_primary_stream)
    : _program(program)
    , _config(config)
    , _engine(program->get_engine())
    , _stream(stream)
    , _memory_pool(new memory_pool(program->get_engine()))
    , _internal(is_internal)
    , _is_primary_stream(is_primary_stream)
    , _enable_profiling(config.get_property(ov::enable_profiling))
    , _reset_arguments(true) {
    if (!_internal) {
        net_id = get_unique_net_id();
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->after_proc.size() != 0) {
        wait_for_the_turn();
    }

    allocate_primitives();
    configure_primitives_second_output();
    check_names();
    build_insts_deps();
    build_exec_order();
    validate_primitives();
    add_default_output_chains();

    if (is_dynamic()) {
        GPU_DEBUG_DEFINE_MEM_LOGGER("dynamic_network_initialization");
        _kernels_cache = std::unique_ptr<kernels_cache>(new kernels_cache(program->get_engine(),
                                                                          program->get_config(),
                                                                          program->get_id(),
                                                                          program->get_task_executor(),
                                                                          kernel_selector::KernelBase::get_db().get_batch_header_str()));
        _impls_cache = std::unique_ptr<ImplementationsCache>(new ImplementationsCache(_impls_cache_capacity));
        _in_mem_kernels_cache = std::unique_ptr<KernelsCache>(new KernelsCache(_in_mem_kernels_cache_capacity));
        _compilation_context = ICompilationContext::create(program->get_engine(), program->get_config(), program->get_id());
    }
}

network::network(engine& engine,
                 const topology& topo,
                 const ExecutionConfig& config,
                 bool is_internal)
    : network(program::build_program(engine, topo, config, is_internal), config, engine.create_stream(config), is_internal) {}

network::network(engine& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const ExecutionConfig& config,
                 std::shared_ptr<InferenceEngine::CPUStreamsExecutor> task_executor,
                 bool is_internal)
    : network(program::build_program(engine, nodes, config, task_executor, is_internal), config, engine.create_stream(config), is_internal) {}

network::network(program::ptr program, uint16_t stream_id)
    : network(program, program->get_config(), program->get_engine().create_stream(program->get_config()), false, stream_id == 0) {}

network::network(program::ptr program, stream::ptr stream, uint16_t stream_id)
    : network(program, program->get_config(), stream, false, stream_id == 0) {}

network::network(cldnn::BinaryInputBuffer& ib, stream::ptr stream, engine& engine, uint16_t stream_id)
    : network(ib, ExecutionConfig{}, stream, engine, stream_id) {}

network::network(cldnn::BinaryInputBuffer& ib, const ExecutionConfig& config, stream::ptr stream, engine& engine, uint16_t stream_id)
    : _program(nullptr)
    , _config(config)
    , _engine(engine)
    , _stream(stream)
    , _memory_pool(new memory_pool(engine))
    , _internal(false)
    , _is_primary_stream(false)
    , _reset_arguments(true) {
    net_id = get_unique_net_id();

    kernels_cache kernels_cache(get_engine(), config, 0, nullptr, {""});
    ib >> kernels_cache;

    int num_data_nodes;
    ib >> num_data_nodes;

    for (int i = 0; i < num_data_nodes; ++i) {
        std::string type;
        std::string _primitive_id;
        ib >> type >> _primitive_id;
        std::shared_ptr<cldnn::primitive_inst> new_primitive_inst = prim_map_storage::instance().get_type_id(type)->create_instance(*this);
        ib >> *new_primitive_inst;
        _primitives[_primitive_id] = new_primitive_inst;
    }

    int exec_order_size;
    ib >> exec_order_size;
    _exec_order.clear();

    std::vector<std::string> _exec_order_types;
    _exec_order_types.resize(exec_order_size);

    for (auto& type : _exec_order_types) {
        ib >> type;
        std::shared_ptr<cldnn::primitive_inst> new_primitive_inst = prim_map_storage::instance().get_type_id(type)->create_instance(*this);
        _exec_order.emplace_back(new_primitive_inst);
    }

    _outputs.clear();
    _output_chains.clear();

    for (const auto& p_inst : _exec_order) {
        ib >> *p_inst;
        _primitives[p_inst->id()] = p_inst;
        p_inst->init_kernels(kernels_cache);
    }

    for (auto& item : _primitives) {
        auto& p_inst = item.second;
        if (p_inst->is_input())
            _inputs.push_back(p_inst);
        if (p_inst->is_output()) {
            _outputs.push_back(p_inst);
            if (p_inst->type() == cldnn::data::type_id())
                _data_outputs.push_back(p_inst);
        }
    }

    for (auto p_inst : _exec_order) {
        p_inst->rebuild_deps(_primitives);
        p_inst->rebuild_exec_deps(_primitives);

        if (p_inst->type() == cldnn::concatenation::type_id() && p_inst->can_be_optimized()) {
            // implicit concat
            std::list<const std::vector<std::pair<std::shared_ptr<const primitive_inst>, int32_t>>*> stack = {&p_inst->dependencies()};
            while (!stack.empty()) {
                auto nodes_list = stack.front();
                stack.pop_front();

                for (auto processed_nodes : *nodes_list) {
                    auto processed_node = processed_nodes.first;
                    auto dep_node = _primitives[processed_node->id()];
                    dep_node->set_output_memory(p_inst->output_memory_ptr(), false);
                    if (processed_node->type() == concatenation::type_id() && processed_node->can_be_optimized()) {
                        if (!processed_node->dependencies().empty())
                            stack.push_back(&processed_node->dependencies());
                    }
                }
            }
        }
    }

    std::map<std::string, std::string> reuse_map;
    ib >> reuse_map;

    for (auto reuse_pair : reuse_map) {
        auto& eltw_inst = _primitives.at(reuse_pair.second);
        auto& prim_inst = _primitives.at(reuse_pair.first);
        auto& eltw_mem = eltw_inst->output_memory();
        auto new_mem = eltw_mem.get_engine()->reinterpret_buffer(eltw_mem, prim_inst->output_memory_ptr()->get_layout());
        prim_inst->set_output_memory(new_mem);
    }

    size_t num_variable_state_primitives;
    ib >> num_variable_state_primitives;
    for (size_t i = 0; i < num_variable_state_primitives; i++) {
        primitive_id p_inst_id;
        ib >> p_inst_id;
        _variable_state_primitives.emplace_back(_primitives.at(p_inst_id));
    }

    add_default_output_chains();
}

network::~network() {
    if (_compilation_context)
        _compilation_context->cancel();
    _memory_pool->clear_pool_for_network(net_id);
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
        dump_perf_data_raw(debug_config->dump_profiling_data + "/perf_raw" + std::to_string(net_id) + ".csv", _exec_order);
    }
}

// Cache blob format:
//     [ cldnn::kernels_cache ]
//     [ non executable primitive_inst ]
//     [ executable primitive_inst ]
//     [ memory reuse information ]
void network::save(cldnn::BinaryOutputBuffer& ob) {
    kernels_cache kernels_cache(get_engine(), _config, 0, nullptr, {""});
    for (const auto& p_inst : _exec_order) {
        if (p_inst->get_impl() != nullptr)
            kernels_cache.add_kernels(p_inst->get_impl()->get_kernel_ids(), p_inst->get_impl()->get_kernels());
    }
    ob << kernels_cache;

    int num_data_nodes = 0;
    for (const auto& p_inst : _primitives) {
        if (p_inst.second->type() == cldnn::data::type_id() ||
           (p_inst.second->type() == cldnn::mutable_data::type_id() && p_inst.second->get_impl() == nullptr)) {
            num_data_nodes += 1;
        }
    }
    ob << num_data_nodes;

    for (const auto& p_inst : _primitives) {
        if (p_inst.second->type() == cldnn::data::type_id() ||
           (p_inst.second->type() == cldnn::mutable_data::type_id() && p_inst.second->get_impl() == nullptr)) {
            ob << p_inst.second->get_node().get_primitive()->type_string();
            ob << p_inst.second->id();
            ob << *(p_inst.second);
        }
    }

    int exec_order_size;
    exec_order_size = _exec_order.size();
    ob << exec_order_size;

    for (const auto& p_inst : _exec_order) {
        ob << p_inst->get_node().get_primitive()->type_string();
    }

    for (const auto& p_inst : _exec_order) {
        ob << *p_inst;
    }

    std::map<std::string, std::string> reuse_map;

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
                    eltw_dep = fused_op.dep_start_idx;
                    auto& eltw_in = node->get_dependency(eltw_dep);
                    if (_primitives.find(eltw_in.id()) != _primitives.end() && _primitives.find(node->id()) != _primitives.end()) {
                        reuse_map[node->id()] = eltw_in.id();
                    }
                }
            }
        }
    }

    ob << reuse_map;

    ob << _variable_state_primitives.size();
    for (const auto& p_inst : _variable_state_primitives) {
        ob << p_inst->id();
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
                                    bool is_internal) {
    return std::make_shared<network>(engine, topology, config, is_internal);
}

network::ptr network::build_network(engine& engine,
                                    const std::set<std::shared_ptr<program_node>>& nodes,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<InferenceEngine::CPUStreamsExecutor> task_executor,
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
        if (!prim->is_dynamic())
            prim->set_arguments();
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
    _events.clear();
}

void network::set_input_data(const primitive_id& id, memory::ptr data) {
    std::shared_ptr<primitive_inst> primitive_inst;

    primitive_inst = find_primitive(id);

    if (primitive_inst == nullptr)
        throw std::runtime_error("topology doesn't contain primitive:" + id);

    if (primitive_inst->type() != input_layout::type_id()) {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }

    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);

    // Wait for previous execution completion
    reset_execution(true);
    input->set_data(data);
}

void network::add_default_output_chains() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("add_default_output_chains");
    for (auto& output : _outputs) {
        add_output_chain(output);
    }
}

network::output_chains_map::iterator network::add_output_chain(std::shared_ptr<primitive_inst>& p_inst) {
    std::vector<std::shared_ptr<primitive_inst>> chain;
    std::stack<std::shared_ptr<const primitive_inst>> candidates;
    auto& eng = get_engine();
    const auto& mem_orig = p_inst->output_memory();

    auto add_mdata_chain = [&](std::shared_ptr<primitive_inst>& p_inst) {
        auto mdata_ptr = std::dynamic_pointer_cast<mutable_data_inst>(p_inst);
        if (!mdata_ptr)
            return;
        // special handling for mutable data, which can share
        // its attached memory with both its inputs and outputs
        for (auto& dep : p_inst->dependencies()) {
            // check dependencies
            if (eng.is_the_same_buffer(mem_orig, dep.first->output_memory())) {
                chain.push_back(std::const_pointer_cast<primitive_inst>(dep.first));
            }
            // then second order dependencies
            for (auto& second_dep : dep.first->dependencies()) {
                if (eng.is_the_same_buffer(mem_orig, second_dep.first->output_memory())) {
                    chain.push_back(std::const_pointer_cast<primitive_inst>(second_dep.first));
                }
            }
        }

        //then users
        const auto& user_ids = mdata_ptr->get_user_ids();
        for (const auto& id : user_ids) {
            auto usr_prim = get_primitive(id);
            if (eng.is_the_same_buffer(mem_orig, usr_prim->output_memory())) {
                chain.push_back(usr_prim);
            }
        }
    };

    if (p_inst->can_be_optimized()) {
        candidates.push(p_inst);
    } else {
        chain.push_back(p_inst);
    }
    add_mdata_chain(p_inst);

    // find all dependencies that are 'optimized'
    while (!candidates.empty()) {
        auto cand = candidates.top();
        candidates.pop();
        const auto& mem_cand = cand->output_memory();
        if (eng.is_the_same_buffer(mem_orig, mem_cand)) {
            auto nc_cand = std::const_pointer_cast<primitive_inst>(cand);
            chain.push_back(nc_cand);
            add_mdata_chain(nc_cand);
        }

        for (auto& dep : cand->dependencies()) {
            if (dep.first->can_be_optimized()) {
                candidates.push(dep.first);
            } else {
                const auto& mem_dep = dep.first->output_memory();
                if (eng.is_the_same_buffer(mem_orig, mem_dep)) {
                    auto nc_dep = std::const_pointer_cast<primitive_inst>(dep.first);
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

void network::set_output_memory(const primitive_id& id, memory::ptr mem_new) {
    std::shared_ptr<primitive_inst> p_inst;

    p_inst = find_primitive(id);

    if (!p_inst)
        throw std::runtime_error("topology doesn't contain primitive: " + id);

    auto iter = std::find(_outputs.begin(), _outputs.end(), p_inst);
    if (iter == _outputs.end())
        throw std::runtime_error("primitive: " + id + " is not a network output");

    // Wait for previous execution completion
    reset_execution(true);

    auto& eng = get_engine();
    // locate primitive chain for this output
    // if no chain found - add it
    auto o_iter = _output_chains.find(id);
    if (o_iter == _output_chains.end()) {
        o_iter = add_output_chain(p_inst);
    }

    for (auto& prim : o_iter->second) {
        prim->set_output_memory(eng.reinterpret_buffer(*mem_new, prim->output_memory().get_layout()), false);
        if (!_reset_arguments &&
            (prim->type() != cldnn::data::type_id() && !(prim->type() == cldnn::mutable_data::type_id() && prim->dependencies().empty()))) {
            prim->set_arguments();
        }
    }
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

    OPENVINO_ASSERT(prim_inst, "[GPU] Can't get implementation type, since topology",
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
    std::vector<std::shared_ptr<program_node>> nodes_to_allocate{};
    auto& po = _program->get_processing_order();
    for (auto node : po) {
        nodes_to_allocate.push_back(_program->get_node_ptr(node->id()));
    }

    std::sort(nodes_to_allocate.begin(),
              nodes_to_allocate.end(),
              [&po](std::shared_ptr<program_node> const& lhs, std::shared_ptr<program_node> const& rhs) {
                    auto lhs_layout = lhs->get_output_layout();
                    auto rhs_layout = rhs->get_output_layout();
                    if (lhs_layout.is_dynamic() && lhs_layout.has_upper_bound()) {
                        lhs_layout.set_tensor(lhs_layout.get_tensor());
                    }
                    if (rhs_layout.is_dynamic() && rhs_layout.has_upper_bound()) {
                        rhs_layout.set_tensor(rhs_layout.get_tensor());
                    }

                    if (rhs_layout.is_dynamic() && !rhs_layout.has_upper_bound() && lhs_layout.is_dynamic() && !lhs_layout.has_upper_bound()) {
                        return po.get_processing_number(lhs.get()) < po.get_processing_number(rhs.get());
                    }

                    if (rhs_layout.is_dynamic())
                        return true;
                    if (lhs_layout.is_dynamic())
                        return false;

                    return (lhs_layout.bytes_count() > rhs_layout.bytes_count());
              });

    for (auto const& node : nodes_to_allocate) {
        allocate_primitive_instance(*node);
    }

    for (auto const& node : po) {
        if (node->get_preferred_impl_type() == impl_types::onednn) {
            size_t eltw_dep = 0;
            for (auto& fused_op : node->get_fused_primitives()) {
                if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                    // If it is first sum, reuse the buffer
                    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(*node, fused_op);
                    if (fusing_type != add_fusing_type::sum || eltw_dep != 0)
                        continue;
                    eltw_dep = fused_op.dep_start_idx;
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
    }
}

void network::build_exec_order() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("build_exec_order");
    for (auto& node : _program->get_processing_order()) {
        if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
            add_to_exec_order(node->id());
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
    // Wait for previous execution completion
    reset_execution(false);
    GPU_DEBUG_TRACE << "----------------------------------------------" << std::endl;
    GPU_DEBUG_TRACE << "Start network execution" << std::endl;

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
    GPU_DEBUG_GET_INSTANCE(debug_config);
    for (auto& inst : _exec_order) {
        GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
            const std::string layer_name = inst->id();
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                std::cerr << inst->id() << std::endl;
            }

            GPU_DEBUG_IF(debug_config->dump_layers_dst_only == 0 &&
                            debug_config->is_dumped_layer(layer_name)) {
                for (size_t i = 0; i < get_primitive(inst->id())->dependencies().size(); i++) {
                    log_memory_to_file(get_primitive(inst->id())->dep_memory_ptr(i), get_stream(),
                                       layer_name + "_src_" + std::to_string(i));
                }
            }
        }

        execute_primitive(inst, events);

        GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
            get_stream().finish();
            const std::string layer_name = inst->id();
            GPU_DEBUG_IF(debug_config->is_dumped_layer(layer_name, inst->is_output())) {
                for (size_t i = 0; i < get_primitive(inst->id())->outputs_memory_count(); i++) {
                    log_memory_to_file(get_primitive(inst->id())->output_memory_ptr(i), get_stream(),
                                       layer_name + "_dst_" + std::to_string(i));
                }
            }
        }
    }

    // Store events only in case of OOO queue or enabled Profiling
    auto store_events = get_stream().get_queue_type() == QueueTypes::out_of_order || _enable_profiling;
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
    return _program->get_primitives_info();
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

std::vector<std::shared_ptr<primitive_inst>> network::get_primitives(const std::vector<primitive_id>& ids) {
    std::vector<std::shared_ptr<primitive_inst>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) {
        return get_primitive(id);
    });
    return result;
}

std::vector<std::pair<std::shared_ptr<primitive_inst>, int>> network::get_primitives(const std::vector<std::pair<program_node*, int>>& nodes) {
    std::vector<std::pair<std::shared_ptr<primitive_inst>, int>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const std::pair<program_node*, int>& node) {
        return std::make_pair(get_primitive(node.first->id()), node.second);
    });
    return result;
}

void network::execute_primitive(const std::shared_ptr<primitive_inst>& primitive,
                                const std::vector<event::ptr>& events) {
    event::ptr ev = primitive->execute(events);

    // Collect events only for OOO queue and Profiling mode
    if (get_stream().get_queue_type() == QueueTypes::out_of_order || _enable_profiling) {
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
                if (dep.first->is_type<input_layout>() || dep.first->is_type<mutable_data>()) {
                    return true;
            }
            if (dep.first->can_be_optimized()) {
                if (is_mutable_input(*dep.first)) {
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
    if (std::dynamic_pointer_cast<assign_inst>(inst) || std::dynamic_pointer_cast<read_value_inst>(inst))
        _variable_state_primitives.push_back(inst);
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

memory::ptr network::get_memory_from_pool(const layout& layout,
                                               primitive_id id,
                                               std::set<primitive_id> dependencies,
                                               allocation_type type,
                                               bool reusable) {
    if (_config.get_property(ov::intel_gpu::enable_memory_pool))
        return _memory_pool->get_memory(layout, id, get_id(), dependencies, type, reusable);
    return _memory_pool->get_memory(layout, type);
}

network::VariableState& network::get_variable_memory(const std::string &variable_id) {
    auto it = _variables_states.find(variable_id);
    if (it == _variables_states.end()) {
        CLDNN_ERROR_MESSAGE(variable_id, "Variable not found");
    }
    return *it->second;
}

void network::assign_variables_memories(variables_states_map &&variables_memories) {
    _variables_states = variables_memories;
    for (auto primitive : _variable_state_primitives) {
        if (const auto& memory_state_primitive = std::dynamic_pointer_cast<memory_state::variable>(primitive)) {
            auto it = _variables_states.find(memory_state_primitive->variable_id());
            if (it != _variables_states.end())
                primitive->set_output_memory(it->second->memory, false);
            else
                CLDNN_ERROR_MESSAGE(memory_state_primitive->variable_id(), "Memory state not found");
        }
    }
}

}  // namespace cldnn
