// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "json_object.h"
#include "primitive_type_base.h"
#include "recurrent_cache_update_inst.h"
#include "reorder_inst.h"
namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(recurrent_cache_update)
layout recurrent_cache_update_inst::calc_output_layout(const recurrent_cache_update_node& n, const kernel_impl_params& p) {
    return calc_output_layouts<ov::PartialShape>(n, p)[0];
}
template <typename ShapeType>
std::vector<layout> recurrent_cache_update_inst::calc_output_layouts(const recurrent_cache_update_node&, const kernel_impl_params& p) {
    const auto d = p.typed_desc<recurrent_cache_update>();
    const bool conv = d->history > 0;
    const auto& data = p.get_input_layout(conv ? 6 : 2);
    auto s = data.get_partial_shape();
    if (conv)
        s = {s[0], s[1], s[3], s[2]};
    const auto& cache = p.get_input_layout(conv ? 0 : 3);
    return {layout(s, data.data_type, format::bfyx), layout(cache.get_partial_shape(), cache.data_type, format::bfyx)};
}
template std::vector<layout> recurrent_cache_update_inst::calc_output_layouts<ov::PartialShape>(const recurrent_cache_update_node&, const kernel_impl_params&);
std::string recurrent_cache_update_inst::to_string(const recurrent_cache_update_node& n) {
    std::stringstream s;
    n.desc_to_json()->dump(s);
    return s.str();
}
recurrent_cache_update_inst::typed_primitive_inst(network& net, const recurrent_cache_update_node& n) : parent(net, n) {}

bool recurrent_cache_update_inst::try_bind_output_memory(const layout& actual_layout, size_t idx) {
    if (idx != 1 || !is_dynamic())
        return false;
    if (try_bind_cache_output(actual_layout))
        return true;

    // A rejected or removed binding must fall back to private output storage.
    if (_remote_cache_output_alias) {
        if (_outputs[idx] && get_network().get_engine().is_the_same_buffer(*_outputs[idx], *_remote_cache_output_alias)) {
            _outputs[idx].reset();
            _max_output_layout_count[idx] = 0;
            set_flag(ExecutionFlags::MEMORY_CHANGED);
        }
        _remote_cache_output_alias.reset();
    }
    return false;
}

bool recurrent_cache_update_inst::try_bind_cache_output(const layout& actual_layout) {
    constexpr size_t idx = 1;

    for (auto* user : get_user_insts()) {
        if (!user->is_output() || !user->get_node().is_type<reorder>() || !user->can_be_optimized() || user->dependencies().size() != 1 ||
            user->dependencies().front().first != this || user->dependencies().front().second != static_cast<int32_t>(idx) ||
            !get_network().has_output_remote_memory_ptr(user->id()))
            continue;
        auto remote = user->output_memory_ptr();
        if (!remote || !get_network().get_engine().is_the_same_buffer(*remote, get_network().get_output_remote_memory(user->id())))
            continue;
        // Infer the Result reorder layout without mutating another primitive.
        // Its normal preparation will update its shape after this producer.
        auto result_params = *user->get_impl_params();
        result_params.input_layouts[0] = actual_layout;
        const auto target_layout = user->get_node().type()->calc_output_layouts(user->get_node(), result_params)[0];
        if ((remote->get_allocation_type() != allocation_type::usm_device && remote->get_allocation_type() != allocation_type::usm_host) ||
            !target_layout.is_static() || actual_layout.data_type != target_layout.data_type || actual_layout.format != target_layout.format ||
            actual_layout.get_pitches() != target_layout.get_pitches() || actual_layout.get_linear_offset() != target_layout.get_linear_offset() ||
            actual_layout.bytes_count() > remote->size())
            continue;
        const size_t cache_input = get_node().as<recurrent_cache_update>().get_primitive()->history == 0 ? 3 : 0;
        bool alias_cache = get_network().get_engine().is_the_same_buffer(*input_memory_ptr(cache_input), *remote);
        const primitive_inst* cache_root = dependencies().at(cache_input).first;
        if (alias_cache) {
            const auto& source_layout = input_memory_ptr(cache_input)->get_layout();
            alias_cache = source_layout.data_type == actual_layout.data_type && source_layout.format == actual_layout.format &&
                          source_layout.get_pitches() == actual_layout.get_pitches() && source_layout.get_linear_offset() == actual_layout.get_linear_offset();
            // Only this new op may consume the original cache. Other graph readers retain
            // immutable semantics; partial overlaps and unrelated input aliases fall back.
            // llama.cpp views introduce optimized Reshapes between Parameter and
            // cache input. Follow only exclusive, memory-identical alias steps;
            // checking the immediate producer alone misses readers at the root.
            const primitive_inst* consumer = this;
            while (alias_cache) {
                for (auto* reader : cache_root->get_user_insts()) {
                    if (reader != consumer) {
                        alias_cache = false;
                        break;
                    }
                }
                if (!alias_cache || !cache_root->can_be_optimized() || cache_root->dependencies().size() != 1)
                    break;
                auto* source = cache_root->dependencies().front().first;
                if (!source->output_memory_ptr() || !get_network().get_engine().is_the_same_buffer(*source->output_memory_ptr(), *remote))
                    break;
                consumer = cache_root;
                cache_root = source;
            }
        }
        bool overlap = false;
        for (size_t input = 0; input < inputs_memory_count(); ++input) {
            auto memory = input_memory_ptr(input);
            if (alias_cache && input == cache_input)
                continue;
            if (!memory || !memory->buffer_ptr())
                continue;
            const auto source = reinterpret_cast<uintptr_t>(memory->buffer_ptr());
            const auto destination = reinterpret_cast<uintptr_t>(remote->buffer_ptr());
            overlap |= source <= destination ? destination - source < memory->size() : source - destination < actual_layout.bytes_count();
        }
        // A caller can also bind this Result over an unrelated graph input.
        // Its independent readers may execute after this kernel on another branch.
        if (!overlap) {
            for (const auto& input_id : get_network().get_input_ids()) {
                auto memory = get_network().get_primitive(input_id)->output_memory_ptr();
                if (alias_cache && get_network().get_primitive(input_id).get() == cache_root)
                    continue;
                if (!memory || !memory->buffer_ptr())
                    continue;
                const auto source = reinterpret_cast<uintptr_t>(memory->buffer_ptr());
                const auto destination = reinterpret_cast<uintptr_t>(remote->buffer_ptr());
                if (source <= destination ? destination - source < memory->size() : source - destination < actual_layout.bytes_count()) {
                    overlap = true;
                    break;
                }
            }
        }
        // Rejected aliases retain private output and a final Result copy.
        if (overlap)
            continue;
        auto& engine = get_network().get_engine();
        const bool changed = !_outputs[idx] || !engine.is_the_same_buffer(*_outputs[idx], *remote);
        if (changed && _outputs[idx] && !_remote_cache_output_alias && get_node().get_program().get_config().get_enable_memory_pool()) {
            get_network().get_memory_pool().release_memory(_outputs[idx].get(), get_node().get_unique_id(), get_node().id(), get_network_id());
        }
        _outputs[idx] = engine.reinterpret_buffer(*remote, actual_layout);
        _remote_cache_output_alias = _outputs[idx];
        _max_output_layout_count[idx] = remote->size() / data_type_traits::size_of(actual_layout.data_type);
        if (changed)
            set_flag(ExecutionFlags::MEMORY_CHANGED);
        return true;
    }
    return false;
}

}  // namespace cldnn
