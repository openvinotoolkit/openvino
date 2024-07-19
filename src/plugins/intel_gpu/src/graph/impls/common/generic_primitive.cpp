// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_primitive_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"

#include <vector>

namespace cldnn {
namespace common {

struct generic_primitive_impl : typed_primitive_impl<generic_primitive> {
    using parent = typed_primitive_impl<generic_primitive>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::generic_primitive_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_primitive_impl>(*this);
    }

    generic_primitive_impl() : parent() {}

    explicit generic_primitive_impl(const generic_primitive_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, generic_primitive_inst& instance) override {
        std::vector<memory::ptr> inputs;
        inputs.reserve(instance.inputs_memory_count());
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            inputs.push_back(instance.input_memory_ptr(i));
        }

        std::vector<memory::ptr> outputs;
        outputs.reserve(instance.outputs_memory_count());
        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            outputs.push_back(instance.output_memory_ptr(i));
        }

        return instance.node->get_primitive()->execute_f(events, instance.get_network().get_stream(), inputs, outputs);
    }

    static std::unique_ptr<primitive_impl> create(const generic_primitive_node& arg, const kernel_impl_params&) {
        return make_unique<generic_primitive_impl>(arg);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }
};

namespace detail {

attach_generic_primitive_common::attach_generic_primitive_common() {
    implementation_map<generic_primitive>::add(impl_types::common,
                                               shape_types::dynamic_shape,
                                               generic_primitive_impl::create,
                                               {},
                                               {});
    implementation_map<generic_primitive>::add(impl_types::common, generic_primitive_impl::create, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::generic_primitive_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::generic_primitive)
