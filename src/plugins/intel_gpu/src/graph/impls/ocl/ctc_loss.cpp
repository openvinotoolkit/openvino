// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_ocl_base.hpp"

#include "ctc_loss.hpp"
#include "ctc_loss_inst.hpp"
#include "kernel_base.hpp"

namespace cldnn {
namespace ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class CTCLossGenerator : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    CTCLossGenerator() : SingleKernelGenerator("ctc_loss_ref") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = make_base_jit_constants(node, params);
        const auto& desc = node.as<ctc_loss>().get_primitive();

        jit_constants.add({
            make_jit_constant("PREPROCESS_COLLAPSE_REPEATED", desc->preprocess_collapse_repeated),
            make_jit_constant("CTC_MERGE_REPEATED", desc->ctc_merge_repeated),
            make_jit_constant("UNIQUE", desc->unique),
        });

        return jit_constants;
    }

    WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const override {
        WorkGroupSizes dispatch_data;
        const auto& output = params.output_layouts[0];

        dispatch_data.global = {output.get_shape()[0], 1, 1};
        dispatch_data.local = {1, 1, 1}; /*GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo)*/;
        return dispatch_data;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> CTCLoss::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<ctc_loss>());
    CTCLossGenerator gen;
    auto kds = gen.get_kernels_data(node, params);
    return cldnn::make_unique<primitive_impl_ocl>(kds, std::string(get_type_info().name) + "::" + gen.get_name());
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ctc_loss)
