// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "arg_max_min/arg_max_min_kernel_selector.h"
#include "arg_max_min/arg_max_min_kernel_base.h"
#include "kernel_runner.h"

namespace cldnn {
namespace ocl {

struct arg_max_min_impl : typed_primitive_impl_ocl<arg_max_min> {
    using parent = typed_primitive_impl_ocl<arg_max_min>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<arg_max_min_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<arg_max_min>& instance, int32_t) const override {
        kernel_arguments_data args = parent::get_arguments(instance, 0);

        if (args.inputs.size() == 3) {
            args.inputs.erase(args.inputs.begin() + 1);  // erase constant input in case of TOP_K
        }

        return args;
    }

public:
    static primitive_impl* create(const arg_max_min_node& arg) {
        const auto& primitive = arg.get_primitive();

        const auto& axis = primitive->axis;
        const auto& top_k = primitive->top_k;
        const auto& out_type = primitive->output_type;
        const auto& sort_type = primitive->sort;
        const auto& with_axis = primitive->with_axis;
        const auto& values_first = primitive->values_first;
        const auto& outputs_num = primitive->input.size() == 3 ? 2 : 1;  // second output passed as input for TOP_K layer

        auto argm_params = get_default_params<kernel_selector::arg_max_min_params>(arg);
        auto argm_optional_params =
            get_default_optional_params<kernel_selector::arg_max_min_optional_params>(arg.get_program());

        argm_params.outputs_num = outputs_num;
        argm_params.topK = top_k;
        if (with_axis) {
            switch (axis) {
                case arg_max_min::batch:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::BATCH;
                    break;
                case arg_max_min::feature:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::FEATURE;
                    break;
                case arg_max_min::x:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::X;
                    break;
                case arg_max_min::y:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::Y;
                    break;
                case arg_max_min::z:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::Z;
                    break;
                default:
                    break;
            }
        }

        if (out_type == primitive->max)
            argm_params.argMaxMinOut = kernel_selector::argm_output::MAX;
        else
            argm_params.argMaxMinOut = kernel_selector::argm_output::MIN;

        if (sort_type == primitive->sort_by_values)
            argm_params.argMaxMinSortType = kernel_selector::argm_sort::VALUE;
        else
            argm_params.argMaxMinSortType = kernel_selector::argm_sort::INDEX;

        if (outputs_num == 2) {
            argm_params.inputs.push_back(convert_data_tensor(arg.get_dependency(2).get_output_layout()));
        }

        argm_params.values_first = values_first;

        auto& kernel_selector = kernel_selector::arg_max_min_kernel_selector::Instance();

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(argm_params, argm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto conv = new arg_max_min_impl(arg, best_kernels[0]);

        return conv;
    }
};

namespace detail {
attach_arg_max_min_impl::attach_arg_max_min_impl() {
    implementation_map<arg_max_min>::add(impl_types::ocl, arg_max_min_impl::create,  {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
    });
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
