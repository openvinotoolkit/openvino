// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reverse_sequence/reverse_sequence_kernel_selector.h"
#include "reverse_sequence/reverse_sequence_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct reverse_sequence_impl : typed_primitive_impl_ocl<reverse_sequence> {
    using parent = typed_primitive_impl_ocl<reverse_sequence>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reverse_sequence_impl>(*this);
    }

public:
    static primitive_impl* create(const reverse_sequence_node& arg) {
        auto reverse_sequence_params = get_default_params<kernel_selector::reverse_sequence_params>(arg);
        auto reverse_sequence_optional_params =
            get_default_optional_params<kernel_selector::reverse_sequence_optional_params>(arg.get_program());

        reverse_sequence_params.seq_axis = arg.get_primitive()->seq_axis;
        reverse_sequence_params.batch_axis = arg.get_primitive()->batch_axis;

        reverse_sequence_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::reverse_sequence_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reverse_sequence_params, reverse_sequence_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reverse_sequence = new reverse_sequence_impl(arg, best_kernels[0]);

        return reverse_sequence;
    }
};

namespace detail {

attach_reverse_sequence_impl::attach_reverse_sequence_impl() {
    implementation_map<reverse_sequence>::add(impl_types::ocl, reverse_sequence_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
