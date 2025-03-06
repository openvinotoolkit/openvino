// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "assign_inst.h"
#include "kv_cache_inst.h"
#include "read_value_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"

namespace cldnn {
namespace cpu {

struct read_value_impl : public typed_primitive_impl<read_value> {
    using parent = typed_primitive_impl<read_value>;
    using parent::parent;

    std::string variable_id;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::read_value_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<read_value_impl>(*this);
    }

    read_value_impl() : parent() {}

    explicit read_value_impl(const read_value_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<read_value>());
        const auto& node = arg.as<read_value>();
        variable_id = node.get_primitive()->variable_id;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << variable_id;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> variable_id;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, read_value_inst& instance) override {
        auto& variable = instance.get_network().get_variable(variable_id);
        auto &stream = instance.get_network().get_stream();
        stream.wait_for_events(events);

        OPENVINO_ASSERT(variable.get_layout() == instance.get_output_layout(),
                "[GPU] Layout mismatch: variable layout: ", variable.get_layout().to_short_string(),
                " read_value output layout: ", instance.get_output_layout().to_short_string());

        if (!variable.is_set()) {
            if (instance.get_impl_params()->input_layouts.size() > 0) {
                variable.get_memory()->copy_from(stream, instance.dep_memory(0), true);
            } else {
                variable.get_memory()->fill(stream);
            }
            if (!instance.get_user_insts().empty()) {
                auto user_inst = instance.get_user_insts().front();
                if (!(user_inst->get_node().is_type<assign>() || user_inst->get_node().is_type<kv_cache>()) &&
                    instance.get_network().contains_state(variable_id)) {
                    variable.set();
                }
            }
        }

        if (!instance.can_be_optimized()) {
            GPU_DEBUG_TRACE_DETAIL << "Copy variable's memory to new read_value's output buffer\n";
            std::vector<cldnn::event::ptr> res_events;
            res_events.push_back(instance.output_memory(0).copy_from(stream, *variable.get_memory(), false));

            if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                auto scales_state = compressed_cache_variable->get_compression_scale_state();
                res_events.push_back(instance.output_memory(1).copy_from(stream, *scales_state->get_memory(), false));

                if (compressed_cache_variable->has_zp_state()) {
                    auto zp_state = compressed_cache_variable->get_compression_zp_state();
                    res_events.push_back(instance.output_memory(1).copy_from(stream, *zp_state->get_memory(), false));
                }
            }

            return stream.aggregate_events(res_events, res_events.size() > 1);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const read_value_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<read_value_impl>(arg);
    }
};

namespace detail {

attach_read_value_impl::attach_read_value_impl() {
    implementation_map<read_value>::add(impl_types::cpu, shape_types::dynamic_shape, read_value_impl::create, {});
    implementation_map<read_value>::add(impl_types::cpu, shape_types::static_shape, read_value_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::read_value_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::read_value)
