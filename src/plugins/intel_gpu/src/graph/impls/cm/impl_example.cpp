// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cm/impl_example.hpp"

#include "fully_connected/cm/fully_connected_cm_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"
#include "fully_connected_inst.h"
#include "impls/ocl/primitive_base.hpp"

namespace cldnn {
namespace cm {

struct example_impl : ocl::typed_primitive_impl_ocl<fully_connected> {
    using parent = typed_primitive_impl_ocl<fully_connected>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::fully_connected_cm_kernel_selector;
    using kernel_params_t = kernel_selector::fully_connected_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cm::example_impl)

    example_impl() = default;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<example_impl, kernel_params_t>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<fully_connected>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<fully_connected>();

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        args.inputs = {instance.input_memory_ptr(0)};
        size_t in_id = instance.bias_term() ? 3 : 2;
        if (!desc->decompression_scale.empty())
            args.inputs.push_back(instance.dep_memory_ptr(in_id++));

        if (!desc->decompression_zero_point.empty())
            args.inputs.push_back(instance.dep_memory_ptr(in_id));

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(impl_param,
                                                                                               false,
                                                                                               is_shape_agnostic);
        return params;
    }
};
std::unique_ptr<primitive_impl> ExampleImplementationManager::create_impl(const program_node& node,
                                                                          const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<fully_connected>());
    return ocl::typed_primitive_impl_ocl<fully_connected>::create<example_impl>(
        static_cast<const fully_connected_node&>(node),
        params);
}
}  // namespace cm
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cm::example_impl)
