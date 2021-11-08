// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gemm/gemm_kernel_selector.h"
#include "gemm/gemm_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct gemm_impl : typed_primitive_impl_ocl<gemm> {
    using parent = typed_primitive_impl_ocl<gemm>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_impl>(*this);
    }

public:
    static primitive_impl* create(const gemm_node& arg) {
        auto gemm_params = get_default_params<kernel_selector::gemm_params>(arg, 1);
        auto gemm_optional_params =
            get_default_optional_params<kernel_selector::gemm_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            gemm_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        auto desc = arg.get_primitive();
        gemm_params.alpha = desc->alpha;
        gemm_params.beta = desc->beta;
        gemm_params.transpose_input0 = desc->transpose_input0;
        gemm_params.transpose_input1 = desc->transpose_input1;

        if (arg.get_output_layout().data_type == data_types::i8 ||
            arg.get_output_layout().data_type == data_types::u8) {
            gemm_params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            gemm_params.quantization = kernel_selector::QuantizationType::NONE;
        }

        auto& kernel_selector = kernel_selector::gemm_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gemm_params, gemm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new gemm_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_gemm_impl::attach_gemm_impl() {
    implementation_map<gemm>::add(impl_types::ocl, gemm_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
