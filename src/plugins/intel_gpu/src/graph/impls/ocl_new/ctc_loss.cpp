// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/ocl_new/utils/dispatch_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_ocl_base.hpp"

#include "ctc_loss.hpp"
#include "ctc_loss_inst.hpp"
#include "utils/kernel_base.hpp"

namespace ov::intel_gpu::ocl {
namespace {

using namespace ov::intel_gpu::ocl;

class CTCLossGenerator : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    CTCLossGenerator() : SingleKernelGenerator("ctc_loss") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<ctc_loss>();

        jit_constants.add({
            make_jit_constant("PREPROCESS_COLLAPSE_REPEATED", desc->preprocess_collapse_repeated),
            make_jit_constant("CTC_MERGE_REPEATED", desc->ctc_merge_repeated),
            make_jit_constant("UNIQUE", desc->unique),
        });

        return jit_constants;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd) {
            WorkGroupSizes wgs;
            const auto& output = params.output_layouts[0];

            wgs.global = {output.get_shape()[0], 1, 1};
            wgs.local = get_optimal_lws(wgs.global, params.get_device_info());
            return { wgs, {} };
        };

        return f;
    }
};


class CTCLossImpl : public PrimitiveImplOCL {
public:
    CTCLossImpl(const program_node& node, const kernel_impl_params& params)
        : PrimitiveImplOCL(std::string(CTCLoss::get_type_info_static().name)) {
        add_stage<CTCLossGenerator, 0>(params);
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<CTCLossImpl>(*this);
    }

    std::vector<layout> get_internal_buffer_layouts(const kernel_impl_params& params) const override {
        return {};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> CTCLoss::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<ctc_loss>());
    return std::make_unique<CTCLossImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ctc_loss)
