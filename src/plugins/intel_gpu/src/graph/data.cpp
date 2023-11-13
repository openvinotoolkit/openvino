// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "data_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"

#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(data)

namespace {
memory::ptr attach_or_copy_data(network& network, memory::ptr mem) {
    auto& engine = network.get_engine();
    if (mem->is_allocated_by(engine))
        return mem;

    memory::ptr result = engine.allocate_memory(mem->get_layout(), false);
    mem_lock<char, mem_lock_type::read> src(mem, network.get_stream());
    mem_lock<char, mem_lock_type::write> dst(result, network.get_stream());
    std::copy(src.begin(), src.end(), dst.begin());
    return result;
}
}  // namespace

data_node::typed_program_node(const std::shared_ptr<data> dprim, program& prog)
    : parent(dprim, prog), mem(dprim->mem) {
    constant = true;
    can_share_buffer(false);
    recalc_output_layout(false);
}

void data_node::attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed) {
    mem = new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

std::string data_inst::to_string(data_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

data_inst::typed_primitive_inst(network& network, data_node const& node)
    : parent(network, node, attach_or_copy_data(network, node.get_attached_memory_ptr())) {}

// Cache blob format:
//     [ kernel_impl_params ]
//     [ output memory information ]
//     [ data stored in memory ]
void data_inst::save(cldnn::BinaryOutputBuffer& ob) const {
    parent::save(ob);
    ob << _outputs[0]->get_layout();

    const auto _allocation_type = _outputs[0]->get_allocation_type();
    ob << make_data(&_allocation_type, sizeof(_allocation_type));

    size_t data_size = _outputs[0]->size();
    ob << make_data(&data_size, sizeof(size_t));

    if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
        ob << make_data(_outputs[0]->buffer_ptr(), data_size);
    } else {
        std::vector<uint8_t> _buf;
        _buf.resize(data_size);
        _outputs[0]->copy_to(get_network().get_stream(), _buf.data());
        ob << make_data(_buf.data(), data_size);
    }
}

void data_inst::load(BinaryInputBuffer& ib) {
    parent::load(ib);
    layout output_layout = layout();
    ib >> output_layout;

    allocation_type _allocation_type = allocation_type::unknown;
    ib >> make_data(&_allocation_type, sizeof(_allocation_type));

    size_t data_size = 0;
    ib >> make_data(&data_size, sizeof(size_t));

    if (!get_network().is_primary_stream()) {
        _outputs[0] = ib.getConstData(_network.get_local_id(), id());
        auto pos = ib.tellg();
        pos += data_size;
        ib.seekg(pos);
    } else {
        _outputs[0] = get_network().get_engine().allocate_memory(output_layout, _allocation_type, false);

        if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
            ib >> make_data(_outputs[0]->buffer_ptr(), data_size);
        } else {
            std::vector<uint8_t> _buf;
            _buf.resize(data_size);
            ib >> make_data(_buf.data(), data_size);
            _outputs[0]->copy_from(get_network().get_stream(), _buf.data());
        }

        ib.addConstData(_network.get_local_id(), id(), _outputs[0]);
    }
}

}  // namespace cldnn
