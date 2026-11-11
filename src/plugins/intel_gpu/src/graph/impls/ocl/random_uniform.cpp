// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "random_uniform_inst.h"
#include "random_uniform/random_uniform_kernel_ref.h"
#include "random_uniform/random_uniform_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct random_uniform_impl : typed_primitive_impl_ocl<random_uniform> {
    using parent = typed_primitive_impl_ocl<random_uniform>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::random_uniform_kernel_selector;
    using kernel_params_t = kernel_selector::random_uniform_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::random_uniform_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<random_uniform_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && !_kernel_data.kernelName.empty()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    // Kernel scalars (Philox state) are updated on the host before every execution, so a recorded command list
    // cannot be replayed: it would reuse the recorded state and produce the same values on every inference.
    bool supports_replay() const override {
        return false;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, random_uniform_inst& instance) override {
        // RandomUniform must produce a new sequence on every execution: pass the current Philox state
        // as runtime scalars, re-set kernel arguments (static graphs set them only once), then advance the state.
        if (!instance.can_be_optimized()) {
            OPENVINO_ASSERT(_kernel_data.kernels.size() == 1 && _kernel_data.kernels[0].params.scalars.size() == 3,
                            "[GPU] RandomUniform kernel expects 3 runtime scalars (global_seed, n_state, counter_state)");
            auto& scalars = _kernel_data.kernels[0].params.scalars;
            scalars[0].v.u64 = instance.get_global_seed();
            scalars[1].v.u64 = instance.get_n_state();
            scalars[2].v.u64 = instance.get_counter_state();
            parent::set_arguments_impl(instance);
        }

        auto ev = parent::execute_impl(events, instance);
        instance.advance_state(instance.get_output_layout().count());
        return ev;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<random_uniform>();
        auto params = get_default_params<kernel_selector::random_uniform_params>(impl_param, is_shape_agnostic);
        params.global_seed = primitive->global_seed;
        params.op_seed = primitive->op_seed;
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_random_uniform_impl::attach_random_uniform_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::bf16,
        data_types::i32,
        data_types::i64
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<random_uniform>::add(impl_types::ocl,
                                            shape_types::static_shape,
                                            typed_primitive_impl_ocl<random_uniform>::create<random_uniform_impl>,
                                            types,
                                            formats);

    implementation_map<random_uniform>::add(impl_types::ocl,
                                            shape_types::dynamic_shape,
                                            typed_primitive_impl_ocl<random_uniform>::create<random_uniform_impl>,
                                            types,
                                            formats);
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::random_uniform_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::random_uniform)
