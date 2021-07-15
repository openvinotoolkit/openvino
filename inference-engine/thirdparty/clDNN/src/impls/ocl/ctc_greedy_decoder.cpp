// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_greedy_decoder_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "ctc_greedy_decoder/ctc_greedy_decoder_kernel_selector.h"
#include "ctc_greedy_decoder/ctc_greedy_decoder_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct ctc_greedy_decoder_impl : typed_primitive_impl_ocl<ctc_greedy_decoder> {
    using parent = typed_primitive_impl_ocl<ctc_greedy_decoder>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<ctc_greedy_decoder_impl>(*this);
    }

public:
    static primitive_impl* create(const ctc_greedy_decoder_node& arg) {
        auto ctc_gd_params = get_default_params<kernel_selector::ctc_greedy_decoder_params>(arg);
        auto ctc_gd_optional_params = get_default_optional_params<kernel_selector::ctc_greedy_decoder_optional_params>(arg.get_program());
        auto prim = arg.get_primitive();

        ctc_gd_params.inputs.push_back(
            convert_data_tensor(arg.seq_indicators().get_output_layout()));
        ctc_gd_params.merge_repeated = prim->ctc_merge_repeated;
        ctc_gd_params.blank_index = prim->blank_index;
        ctc_gd_params.outputs_num = arg.has_second_output() ? 2 : 1;

        if (ctc_gd_params.outputs_num == 2) {
            ctc_gd_params.inputs.push_back(
                convert_data_tensor(arg.second_output().get_output_layout()));
        }

        auto& kernel_selector = kernel_selector::ctc_greedy_decoder_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(
            ctc_gd_params, ctc_gd_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto grn = new ctc_greedy_decoder_impl(arg, best_kernels[0]);

        return grn;
    }
};

namespace detail {

attach_ctc_greedy_decoder_impl::attach_ctc_greedy_decoder_impl() {
    implementation_map<ctc_greedy_decoder>::add(impl_types::ocl, ctc_greedy_decoder_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
