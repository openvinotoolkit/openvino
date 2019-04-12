/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "network_impl.h"
#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"
#include "api/CPP/data.hpp"
#include "api/CPP/mutable_data.hpp"
#include "api/CPP/input_layout.hpp"

#include "error_handler.h"
#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "condition_inst.h"
#include "kernel_selector_helper.h"
#include <algorithm>

#include "gpu/ocl_toolkit.h"


//#define DEBUG_DUMP_PATH "/tmp/dump/"


#ifdef DEBUG_DUMP_PATH
#include <iomanip>
#include <fstream>

#define DUMP_VERBOSE 0
#define DUMP_SINGLE_LAYER 0
#define DUMP_LAYER_NAME ""
#endif

namespace cldnn
{
#ifdef DEBUG_DUMP_PATH
static float convert_half_to_float(half_t val, bool flush_denorm_to_zero = false)
    {
#if defined HALF_HALF_HPP
        return val;
#else
        // FP32 parts extracted from FP16.
        uint32_t sign = (static_cast<uint16_t>(val) & 0x8000U) << 16;
        uint32_t mantissa = (static_cast<uint16_t>(val) & 0x3FFU) << 13;

        uint32_t exp_val_f16 = (static_cast<uint16_t>(val) & 0x7C00U) >> 10;
        uint32_t exp;
        if (exp_val_f16 == 0)
        {
            // Handling +/-0 and denormals.
            if (mantissa == 0)
            {
                exp = 0;
            }
            else if (flush_denorm_to_zero)
            {
                sign = 0;
                exp = 0;
                mantissa = 0;
            }
            else
            {
                // Denorms conversion to normal numbers.
                exp = 127 - 15;
                while (!(mantissa & 0x400000U))
                {
                    mantissa <<= 1;
                    --exp;
                }
                mantissa = (mantissa << 1) & 0x7FFFFFU;
                exp <<= 23;
            }
        }
        else
        {
            // Handling +/-infinity, NaN and normal numbers.
            exp = (exp_val_f16 == 0x1FU ? 0xFFU : exp_val_f16 + 127 - 15) << 23;
        }

        float ret;
        reinterpret_cast<uint32_t&>(ret) = sign | exp | mantissa;

        return ret;
#endif
    }

    float convert_element(float f)
    {
        return f;
    }

    float convert_element(half_t h)
    {
        return convert_half_to_float(h);
    }

    template <class T>
    static void dump(memory_impl& mem, std::ofstream& file_stream)
    {
        auto&& size = mem.get_layout().size;

        file_stream << "shape: ";
        file_stream << size.batch[0] << " ";
        file_stream << size.feature[0] << " ";
        file_stream << size.spatial[1] << " ";
        file_stream << size.spatial[0] << " ";
        file_stream << "(" << size.batch[0] * size.feature[0] * size.spatial[1] * size.spatial[0] << ")" << std::endl;

        auto mem_ptr = static_cast<T*>(mem.lock());

        for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b)
        {
            for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f)
            {
                for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y)
                {
                    for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x)
                    {
                        cldnn::tensor t(cldnn::batch(b), cldnn::feature(f), cldnn::spatial(x, y));
                        size_t input_it = mem.get_layout().get_linear_offset(t);
                        file_stream << std::fixed << std::setprecision(6) << convert_element(mem_ptr[input_it]) << std::endl;
                    }
                }
            }
        }

        mem.unlock();
    }

    static void log_memory_to_file(memory_impl& mem, std::string layerName)
    {
        std::string filename = layerName;
        std::replace(filename.begin(), filename.end(), '\\', '_');
        std::replace(filename.begin(), filename.end(), '/', '_');
        std::replace(filename.begin(), filename.end(), ' ', '_');
        std::replace(filename.begin(), filename.end(), ':', '_');
        filename = DEBUG_DUMP_PATH + filename + ".txt";

        std::ofstream file_stream(filename);
        if (mem.get_layout().data_type == cldnn::data_types::f32)
            dump<float>(mem, file_stream);
        else
            dump<half_t>(mem, file_stream);
    }
#endif
/*
Network_impl will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by propagate_constants opt pass).
*/
network_impl::network_impl(const program_impl& program, bool is_internal)
    : _program(&program)
    , _internal(is_internal)
{
    static std::atomic<uint32_t> id_gen{ 0 };
    if (!_internal)
    {
        net_id = ++id_gen;
    }

    allocate_primitives();
    check_names();
    build_insts_deps();
    build_exec_order();
    validate_primitives();
    _program->dump_memory_pool();
}

network_impl::network_impl(engine_impl& engine, const topology_impl& topo, const build_options& options, bool is_internal)
    : network_impl(*engine.build_program(topo, options, is_internal), is_internal)
{
}

network_impl::network_impl(engine_impl& engine, const std::set<std::shared_ptr<program_node>>& nodes, const build_options& options, bool is_internal)
    : network_impl(*engine.build_program(nodes, options, is_internal), is_internal)
{
}

void network_impl::validate_primitives()
{
    for (auto const& prim : _exec_order)
    {
        bool valid = prim->validate();
        CLDNN_ERROR_NOT_EQUAL(prim->id(), "validate", valid, "", true, "has not a valid instance.");
    }
}

void network_impl::reset_execution(bool wait)
{
    if (wait && _events.size() > 0)
    {
        std::vector<event_impl::ptr> events;
        for (auto& pair : _events)
        {
            auto& ev = pair.second;
            if (ev->is_set())
                continue;

            events.push_back(ev);
        }

        get_engine().wait_for_events(events);
    }
    _events.clear();
}

void network_impl::set_input_data(const primitive_id& id, memory_impl& data)
{
    std::shared_ptr<primitive_inst> primitive_inst;

    primitive_inst = find_primitive(id);

    if(primitive_inst == nullptr)
        throw std::runtime_error("topology doesn't contain prmitive:" + id);

    if (primitive_inst->type() != input_layout::type_id())
    {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }

    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);

    //Wait for previous execution completion
    reset_execution(true);
    input->set_data(data);
}

void cldnn::network_impl::check_names()
{
    for (auto const& prim : _primitives)
    {
        if (find_in_internal_networks(prim.first) != nullptr)
            CLDNN_ERROR_MESSAGE("Network_impl", "Found primitive with id: " + prim.first
                + "in anotother network.");
    }
}

std::shared_ptr<primitive_inst> cldnn::network_impl::find_primitive(const primitive_id& id)
{
    std::shared_ptr<primitive_inst> ret;

    if (_primitives.find(id) != _primitives.end())
        return _primitives.at(id);

    return find_in_internal_networks(id);
}

std::shared_ptr<primitive_inst> cldnn::network_impl::find_in_internal_networks(const primitive_id& id)
{
    std::shared_ptr<primitive_inst> ret;

    for (auto const& prim : _primitives)
    {
        if (prim.second->type() == condition::type_id()) //currently only condition inst contains mini networks
        {
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

void network_impl::set_learning_rate(const float lr)
{
    _learning_rate = lr;
}

float network_impl::get_learning_rate()
{
    return _learning_rate;
}

std::string network_impl::get_primitive_info(const primitive_id& id) const
{
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

void network_impl::allocate_primitives()
{
    std::vector<std::shared_ptr<program_node>> nodes_to_allocate{};
    for (auto node : _program->get_processing_order())
    {
        nodes_to_allocate.push_back(_program->get_node_ptr(node->id()));
    }
    std::sort(nodes_to_allocate.begin(), nodes_to_allocate.end(), [](std::shared_ptr<program_node> const& lhs,
                                                                     std::shared_ptr<program_node> const& rhs)
    {
        return (lhs->get_output_layout().bytes_count() > rhs->get_output_layout().bytes_count());
    });

    for (auto const& node : nodes_to_allocate)
    {
        allocate_primitive_instance(*node);
    }
}

void network_impl::build_insts_deps()
{
    for (auto& inst : _primitives)
    {
        inst.second->build_deps();
    }
}

void network_impl::build_exec_order()
{
    for (auto& node : _program->get_processing_order())
    {
        if (!node->is_type<data>() &&
            !(node->is_type<mutable_data>() && node->get_dependencies().empty()))
        {
            add_to_exec_order(node->id());
        }
    }
}
void network_impl::add_to_exec_order(const primitive_id& id)
{
    auto inst = get_primitive(id);
    _exec_order.push_back(inst);
}

void network_impl::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    //Wait for previous execution completion
    reset_execution(false);

    for (auto& inst : _exec_order)
    {
#ifdef DEBUG_DUMP_PATH
        auto& node = _program->get_node(inst->id());

        std::string layer_name = node.id();
#if DUMP_VERBOSE
        std::cerr << get_primitive_info(inst->id()) << std::endl;
#endif
#if DUMP_SINGLE_LAYER
        if (layer_name == DUMP_LAYER_NAME)
#endif
        {
            std::cerr << "Dump " << layer_name << " layer" << std::endl;
            for (size_t i = 0; i < get_primitive(inst->id())->inputs_memory_count(); i++)
            {
                log_memory_to_file(get_primitive(inst->id())->input_memory(i), layer_name + "_src_" + std::to_string(i));
            }
        }
#endif
        execute_primitive(inst, events);
#ifdef DEBUG_DUMP_PATH
#if DUMP_SINGLE_LAYER
        if (layer_name == DUMP_LAYER_NAME)
#endif
        {
            log_memory_to_file(get_primitive(inst->id())->output_memory(), layer_name + "_dst_0");
        }
        get_engine().flush_network();
#endif
    }

    for (auto& inst : _program->get_processing_order())
    {
        //Special handling for mutable data. The event should be the same as the user or dependency with highest processing_num as
        //the mutable_data can be updated when is both user or dependency.
        if (inst->is_type<mutable_data>())
        {
            decltype(_program->get_processing_order().get_processing_number(inst)) proc_num = 0;
            for (auto& user : inst->get_users())
            {
                auto user_proc_num = _program->get_processing_order().get_processing_number(user);
                if (user_proc_num > proc_num)
                {
                    _events[inst->id()] = _events[user->id()];
                    proc_num = user_proc_num;
                }
            }

            if (!inst->get_dependencies().empty())
            {
                for (auto& dep : inst->get_dependencies())
                {
                    auto dep_proc_num = _program->get_processing_order().get_processing_number(dep);
                    if (dep_proc_num > proc_num)
                    {
                        _events[inst->id()] = _events[dep->id()];
                        proc_num = dep_proc_num;
                    }
                }
            }
        }
    }

    for (auto& dout : _data_outputs) //data primitives are not executed so if they are marked as output we need to add them valid events manually
    {
        _events[dout->id()] = get_engine().create_user_event(true);
    }

    for (auto& prim : _primitives)
    {
        prim.second->reset_output_change();
    }

    get_engine().get_context()->reset_events();

    // Using output of previouse network as input to another one may cause hazard (in OOOQ mode) if user would not
    // provide proper event to execution. Flushing pipeline should prevent this kind of issues.
    // In scenarios with a big number of very small networks it can provide performance drop.
    get_engine().flush_network();
}

std::vector<primitive_id> network_impl::get_output_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_outputs.size());
    for (auto const& output : _outputs)
        ret.push_back(output->id());
    return ret;
}

std::vector<primitive_id> network_impl::get_executed_primitive_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_exec_order.size());
    for (auto const& executed_primitive : _exec_order)
    {
        ret.push_back(executed_primitive->id());
    }
    return ret;
}

std::vector<primitive_id> network_impl::get_all_primitive_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        if(primitive.second->can_be_optimized())
            ret.push_back("_optimized_");
        else
            ret.push_back(primitive.second->id());
    return ret;
}

std::vector<primitive_id> network_impl::get_all_primitive_org_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        ret.push_back(primitive.second->org_id());
    return ret;
}

std::shared_ptr<primitive_inst> network_impl::get_primitive(const primitive_id& id)
{
    if (!_primitives.count(id))
        allocate_primitive_instance(_program->get_node(id));

    return _primitives.at(id);
}

std::vector<std::shared_ptr<primitive_inst>> network_impl::get_primitives(const std::vector<primitive_id>& ids)
{
    std::vector<std::shared_ptr<primitive_inst>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) { return get_primitive(id); });
    return result;
}

std::vector<std::shared_ptr<primitive_inst>> network_impl::get_primitives(const std::vector<program_node*>& nodes)
{
    std::vector<std::shared_ptr<primitive_inst>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const program_node* node) { return get_primitive(node->id()); });
    return result;
}

void network_impl::execute_primitive(const std::shared_ptr<primitive_inst>& primitive, const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    auto id = primitive->id();
    auto it = _events.find(id);
    bool found = (it != _events.end());
    CLDNN_ERROR_BOOL(id, "Invalid primitive call ", found, "Primitive " + id + " is tried to be executed for the second time");

    event_impl::ptr ev;
    if (!get_engine().get_context()->enabled_single_kernel() || get_engine().get_context()->single_kernel_name() == id)
        ev = primitive->execute(events);
    else
        ev = get_engine().create_user_event(true);
    _events.insert({ id, ev });
}

void network_impl::allocate_primitive_instance(program_node const& node)
{
    if (_primitives.count(node.id()))
        return;

    auto inst = node.type()->create_instance(*this, node);
    _primitives[node.id()] = inst;
    if (node.is_input())
        _inputs.push_back(inst);
    if (node.is_output())
    {
        _outputs.push_back(inst);
        if (node.is_type<data>())
            _data_outputs.push_back(inst);
    }
}
}
