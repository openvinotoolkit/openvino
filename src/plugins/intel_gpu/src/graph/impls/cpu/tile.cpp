// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "tile_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/tile.hpp"

namespace cldnn {
namespace cpu {

struct tile_impl : public typed_primitive_impl<tile> {
    using parent = typed_primitive_impl<tile>;
    using parent::parent;

    std::vector<int64_t> repeats;

    std::shared_ptr<ov::op::v0::Tile> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::tile_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<tile_impl>(*this);
    }

    tile_impl() : parent("tile_cpu_impl") {}

    explicit tile_impl(const tile_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<tile>(), "[GPU] Incorrect program_node type");
        repeats = arg.as<tile>().get_primitive()->repeats;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << repeats;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> repeats;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, tile_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "tile::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        if (!op) {
            op = std::make_shared<ov::op::v0::Tile>();
        }

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        if (instance.dependencies().size() == 1) {
            if (repeats.empty())
                OPENVINO_THROW("[GPU] Unexpected configuration of tile impl");

            auto repeats_tensor = ov::Tensor(ov::element::i64, {repeats.size()}, repeats.data());
            input_host_tensors.push_back(repeats_tensor);
        }

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<int32_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);
        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute tile primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const tile_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<tile_impl>();
    }
};


namespace detail {

attach_tile_impl::attach_tile_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<tile>::add(impl_types::cpu, shape_types::static_shape, tile_impl::create, types, formats);
    implementation_map<tile>::add(impl_types::cpu, shape_types::dynamic_shape, tile_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::tile_impl)
