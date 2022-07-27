// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/adaptive_pooling.hpp"
#include "adaptive_pooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"

#include "adaptive_pooling/adaptive_pooling_kernel_selector.h"
#include "adaptive_pooling/adaptive_pooling_kernel_ref.h"


namespace cldnn {
namespace ocl {
struct adaptive_pooling_impl : public typed_primitive_impl_ocl<adaptive_pooling> {
    using parent = typed_primitive_impl_ocl<adaptive_pooling>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<adaptive_pooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<adaptive_pooling>& instance, int32_t) const override {
        kernel_arguments_data args;
        const auto num_inputs = instance.inputs_memory_count();
        for (size_t i = 0; i < num_inputs; ++i) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        args.outputs = {instance.output_memory_ptr()};
        return args;
    }

public:
    static primitive_impl* create(const adaptive_pooling_node& arg) {
        auto params = get_default_params<kernel_selector::adaptive_pooling_params>(arg);
        auto optional_params = get_default_optional_params<kernel_selector::adaptive_pooling_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        if (primitive->mode == adaptive_pooling_mode::average) {
            params.mode = kernel_selector::PoolType::AVG;
        } else {
            params.mode = kernel_selector::PoolType::MAX;

            switch (primitive->index_element_type) {
                case cldnn::data_types::i32: {
                    params.poolIndexElementType = kernel_selector::Datatype::INT32;
                    break;
                }
                case cldnn::data_types::i64: {
                    params.poolIndexElementType = kernel_selector::Datatype::INT64;
                    break;
                }
                default:
                    throw std::runtime_error{"Not supported index element type"};
            }

            params.inputs.push_back(convert_data_tensor(arg.output_indices().get_output_layout()));
        }

        const auto& kernel_selector = kernel_selector::adaptive_pooling_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "best_kernels.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new adaptive_pooling_impl(arg, best_kernels[0]);
    }
};

namespace detail {
attach_adaptive_pooling_impl::attach_adaptive_pooling_impl() {
    implementation_map<adaptive_pooling>::add(impl_types::ocl, adaptive_pooling_impl::create, {
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f16, format::bfzyx),
            std::make_tuple(data_types::f32, format::bfyx),
            std::make_tuple(data_types::f32, format::bfzyx),
            std::make_tuple(data_types::i32, format::bfyx),
            std::make_tuple(data_types::i32, format::bfzyx),
            std::make_tuple(data_types::i64, format::bfyx),
            std::make_tuple(data_types::i64, format::bfzyx),
    });
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
