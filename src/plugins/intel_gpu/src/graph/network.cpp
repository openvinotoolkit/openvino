// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "assign_inst.h"
#include "read_value_inst.h"

#include "to_string_utils.h"
#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "mutable_data_inst.h"
#include "condition_inst.h"
#include "loop_inst.h"
#include "kernel_selector_helper.h"
#include "program_helpers.h"
#include "runtime/cldnn_itt.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <stack>
#include <memory>
#include <set>
#include <utility>
#include <map>
#include <functional>

#ifdef GPU_DEBUG_CONFIG
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#endif

namespace cldnn {

#ifdef GPU_DEBUG_CONFIG
static void dump_perf_data_raw(std::string dump_path, const std::list<std::shared_ptr<primitive_inst>>& exec_order) {
    auto layouts_to_str = [](const std::vector<layout>& layouts) -> std::string {
        std::stringstream s;
        for (size_t i = 0; i < layouts.size(); i++) {
            s << layouts[i].to_short_string();
            if (i != layouts.size() - 1)
                s << ";";
        }
        return s.str();
    };

    const std::string perf_raw_csv_header = "prim_id,prim_type,stage,in_shapes,out_shapes,impl,iters,time_usec\n";
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
                std::string in_l_str = layouts_to_str(key.input_layouts);
                std::string out_l_str = layouts_to_str(key.output_layouts);
                of << prim_id << ","
                << inst->desc()->type_string() << ","
                << key.stage << (key.cache_hit ? " (cache_hit)" : "") << ","
                << in_l_str << ","
                << out_l_str << ","
                << (key.stage == instrumentation::pipeline_stage::inference ? key.impl_name : "undef") << ","
                << num_iters << ","
                << time_avg << "\n";
            }
        }
    }
}

static float convert_half_to_float(half_t val, bool flush_denorm_to_zero = false) {
#if defined HALF_HALF_HPP
    return val;
#else
    // FP32 parts extracted from FP16.
    uint32_t sign = (static_cast<uint16_t>(val) & 0x8000U) << 16;
    uint32_t mantissa = (static_cast<uint16_t>(val) & 0x3FFU) << 13;

    uint32_t exp_val_f16 = (static_cast<uint16_t>(val) & 0x7C00U) >> 10;
    uint32_t exp;
    if (exp_val_f16 == 0) {
        // Handling +/-0 and denormals.
        if (mantissa == 0) {
            exp = 0;
        } else if (flush_denorm_to_zero) {
            sign = 0;
            exp = 0;
            mantissa = 0;
        } else {
            // Denorms conversion to normal numbers.
            exp = 127 - 15;
            while (!(mantissa & 0x400000U)) {
                mantissa <<= 1;
                --exp;
            }
            mantissa = (mantissa << 1) & 0x7FFFFFU;
            exp <<= 23;
        }
    } else {
        // Handling +/-infinity, NaN and normal numbers.
        exp = (exp_val_f16 == 0x1FU ? 0xFFU : exp_val_f16 + 127 - 15) << 23;
    }

    float ret;
    reinterpret_cast<uint32_t&>(ret) = sign | exp | mantissa;

    return ret;
#endif
}

float convert_element(uint32_t u) { return static_cast<float>(u); }

float convert_element(int32_t i) { return static_cast<float>(i); }

float convert_element(float f) { return f; }

float convert_element(half_t h) { return convert_half_to_float(h); }

static size_t get_x_pitch(const layout& layout) {
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
static void dump(memory::ptr mem, stream& stream, std::ofstream& file_stream) {
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

static void log_memory_to_file(memory::ptr mem, stream& stream, std::string layerName) {
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
static void wait_for_the_turn() {
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
static void dump_perf_data_raw(std::string, const std::list<std::shared_ptr<primitive_inst>>&) {}
static void log_memory_to_file(memory::ptr, stream&, std::string) {}
static void wait_for_the_turn() {}
#endif

/*
Network will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by propagate_constants
opt pass).
*/
network::network(program::ptr program, stream::ptr stream, bool is_internal, bool is_primary_stream)
    : _program(program)
    , _stream(stream)
    , _memory_pool(new memory_pool(program->get_engine()))
    , _internal(is_internal)
    , _is_primary_stream(is_primary_stream)
    , _reset_arguments(true) {
    static std::atomic<uint32_t> id_gen{0};
    if (!_internal) {
        net_id = ++id_gen;
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
}

network::network(engine& engine,
                 const topology& topo,
                 const build_options& options,
                 bool is_internal)
    : network(program::build_program(engine, topo, options, is_internal), engine.create_stream(), is_internal) {}

network::network(engine& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const build_options& options,
                 bool is_internal)
    : network(program::build_program(engine, nodes, options, is_internal), engine.create_stream(), is_internal) {}

network::network(program::ptr program, uint16_t stream_id)
    : network(program, program->get_engine().create_stream(), false, stream_id == 0) {}

network::network(program::ptr program, stream::ptr stream, uint16_t stream_id)
    : network(program, stream, false, stream_id == 0) {}

network::~network() {
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
    auto stream = engine.create_stream();
    return std::make_shared<network>(program, stream, is_internal, is_primary_stream);
}

network::ptr network::build_network(engine& engine,
                                    const topology& topology,
                                    const build_options& options,
                                    bool is_internal) {
    return std::make_shared<network>(engine, topology, options, is_internal);
}

network::ptr network::build_network(engine& engine,
                                              const std::set<std::shared_ptr<program_node>>& nodes,
                                              const build_options& options,
                                              bool is_internal) {
    return std::make_shared<network>(engine, nodes, options, is_internal);
}

void network::validate_primitives() {
    for (auto const& prim : _exec_order) {
        bool valid = prim->validate();
        CLDNN_ERROR_NOT_EQUAL(prim->id(), "validate", valid, "", true, "has not a valid instance.");
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
    if (wait && _events.size() > 0) {
        std::vector<event::ptr> events;
        for (auto& pair : _events) {
            auto& ev = pair.second;
            if (ev->is_set())
                continue;

            events.push_back(ev);
        }

        get_stream().wait_for_events(events);
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
            if (eng.is_the_same_buffer(mem_orig, dep->output_memory())) {
                chain.push_back(std::const_pointer_cast<primitive_inst>(dep));
            }
            // then second order dependencies
            for (auto& second_dep : dep->dependencies()) {
                if (eng.is_the_same_buffer(mem_orig, second_dep->output_memory())) {
                    chain.push_back(std::const_pointer_cast<primitive_inst>(second_dep));
                }
            }
        }

        //then users
        const auto& users = p_inst->get_users();
        for (const auto& usr : users) {
            auto usr_prim = get_primitive(usr->id());
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
            if (dep->can_be_optimized()) {
                candidates.push(dep);
            } else {
                const auto& mem_dep = dep->output_memory();
                if (eng.is_the_same_buffer(mem_orig, mem_dep)) {
                    auto nc_dep = std::const_pointer_cast<primitive_inst>(dep);
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
            (!prim->get_node().is_type<data>() && !(prim->get_node().is_type<mutable_data>() && prim->get_node().get_dependencies().empty()))) {
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

std::shared_ptr<primitive_inst> cldnn::network::find_primitive(const primitive_id& id) {
    std::shared_ptr<primitive_inst> ret;

    if (_primitives.find(id) != _primitives.end())
        return _primitives.at(id);

    return find_in_internal_networks(id);
}

std::shared_ptr<primitive_inst> cldnn::network::find_in_internal_networks(const primitive_id& id) {
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

std::string network::get_implementation_info(const primitive_id& id) const {
    return _program->get_implementation_info(id);
}

memory::ptr network::get_output_memory(const primitive_id& output_id) {
    return get_primitive(output_id)->output_memory_ptr();
}

void network::allocate_primitives() {
    std::vector<std::shared_ptr<program_node>> nodes_to_allocate{};
    auto& po = _program->get_processing_order();
    for (auto node : po) {
        nodes_to_allocate.push_back(_program->get_node_ptr(node->id()));
    }

    std::sort(nodes_to_allocate.begin(),
              nodes_to_allocate.end(),
              [&po](std::shared_ptr<program_node> const& lhs, std::shared_ptr<program_node> const& rhs) {
                    if (rhs->get_output_layout().is_dynamic() && lhs->get_output_layout().is_dynamic())
                        return po.get_processing_number(lhs.get()) < po.get_processing_number(rhs.get());
                    if (rhs->get_output_layout().is_dynamic())
                        return true;
                    if (lhs->get_output_layout().is_dynamic())
                        return false;

                    return (lhs->get_output_layout().bytes_count() > rhs->get_output_layout().bytes_count());
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
    // allocate intermediate buffers
    for (auto const& node : po) {
        auto prim = _primitives[node->id()];
        prim->allocate_internal_buffers();
    }
}

void network::configure_primitives_second_output() {
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
    for (auto& inst : _primitives) {
        inst.second->build_deps();
    }
}

void network::build_exec_order() {
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
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "NetworkImpl::Execute");
    // Wait for previous execution completion
    reset_execution(false);
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 1)
        GPU_DEBUG_COUT << "----------------------------------------------" << std::endl;

    std::vector<memory::ptr> in_out_mem;
    bool shared_mem_found = std::any_of(_in_out_shared_mem_types.begin(),
                                        _in_out_shared_mem_types.end(),
                                        [](const shared_mem_type& shared_mem_type) {
                                            return shared_mem_type == shared_mem_type::shared_mem_vasurface ||
                                                   shared_mem_type == shared_mem_type::shared_mem_dxbuffer;
                                        });

    if (shared_mem_found) {
        for (auto& inst : _inputs) {
            if (inst->output_memory_ptr())
                in_out_mem.push_back(inst->output_memory_ptr());
        }

        for (auto& inst : _outputs) {
            if (inst->output_memory_ptr())
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
    for (auto& inst : _exec_order) {
        GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
            auto& node = _program->get_node(inst->id());
            const std::string layer_name = node.id();
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                std::cerr << get_primitive_info(inst->id()) << std::endl;
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
            auto& node = _program->get_node(inst->id());
            const std::string layer_name = node.id();
            GPU_DEBUG_IF(debug_config->is_dumped_layer(layer_name, node.is_output())) {
                log_memory_to_file(get_primitive(inst->id())->output_memory_ptr(), get_stream(), layer_name + "_dst_0");
            }
        }
    }

    // Store events only in case of OOO queue or enabled Profiling
    auto store_events = get_stream().get_queue_type() == queue_types::out_of_order ||
                        get_engine().configuration().enable_profiling;
    if (store_events) {
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
                        auto dep_proc_num = _program->get_processing_order().get_processing_number(dep);
                        if (dep_proc_num > proc_num) {
                            _events[inst->id()] = _events[dep->id()];
                            proc_num = dep_proc_num;
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

std::vector<std::shared_ptr<primitive_inst>> network::get_primitives(const std::vector<primitive_id>& ids) {
    std::vector<std::shared_ptr<primitive_inst>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) {
        return get_primitive(id);
    });
    return result;
}

std::vector<std::shared_ptr<primitive_inst>> network::get_primitives(const std::vector<program_node*>& nodes) {
    std::vector<std::shared_ptr<primitive_inst>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const program_node* node) {
        return get_primitive(node->id());
    });
    return result;
}

void network::execute_primitive(const std::shared_ptr<primitive_inst>& primitive,
                                const std::vector<event::ptr>& events) {
    event::ptr ev = primitive->execute(events);

    // Collect events only for OOO queue and Profiling mode
    if (get_stream().get_queue_type() == queue_types::out_of_order ||
        get_engine().configuration().enable_profiling) {
        auto id = primitive->id();
        _events.insert({id, ev});
    }
}

void network::allocate_primitive_instance(program_node const& node) {
    if (_primitives.count(node.id()))
        return;

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 4) {
        GPU_DEBUG_COUT << node.id() << ": allocate primitive instance" << std::endl;
    }

    auto inst = node.type()->create_instance(*this, node);

    std::function<bool(const program_node&)> is_mutable_input = [&is_mutable_input](const program_node& node) {
        for (auto& dep : node.get_dependencies()) {
                if (dep->is_type<input_layout>() || dep->is_type<mutable_data>()) {
                    return true;
            }
            if (dep->can_be_optimized()) {
                if (is_mutable_input(*dep)) {
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
    if (node.is_input()) {
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
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "NetworkImpl::TransferMemory");
    auto& inst_mem = instance->output_memory();
    auto alloc_type = inst_mem.get_allocation_type();

    // Do not transfer memory if a user requires lockable memory.
    // If memory is used in both gpu and cpu implementations, primitive itself is responsible for correct allocation type
    if (node.need_lockable_memory())
        return;

    if (!get_engine().supports_allocation(allocation_type::usm_device))
        return;

    if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
        // Allocate and transfer memory
        auto device_mem = inst_mem.get_engine()->allocate_memory(inst_mem.get_layout(), allocation_type::usm_device, false);
        device_mem->copy_from(get_stream(), inst_mem);
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << node.id() << ": constant]" << std::endl;
        }
        _memory_pool->release_memory(&inst_mem, node.id(), get_id());
        instance->set_output_memory(device_mem);
    }
}

memory::ptr network::get_memory_from_pool(const layout& layout,
                                               primitive_id id,
                                               std::set<primitive_id> dependencies,
                                               allocation_type type,
                                               bool reusable) {
    if (get_engine().configuration().use_memory_pool)
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
