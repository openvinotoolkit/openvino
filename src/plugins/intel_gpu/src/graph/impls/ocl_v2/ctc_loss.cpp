// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_loss.hpp"

#include "common_utils/dispatch_utils.hpp"
#include "ctc_loss_inst.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class CTCLossGenerator : public KernelGenerator {
public:
    CTCLossGenerator() : KernelGenerator("ctc_loss") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<ctc_loss>();

        jit_constants.add({
            make_jit_constant("PREPROCESS_COLLAPSE_REPEATED", desc->preprocess_collapse_repeated),
            make_jit_constant("CTC_MERGE_REPEATED", desc->ctc_merge_repeated),
            make_jit_constant("UNIQUE", desc->unique),
        });

        return jit_constants;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& output = params.output_layouts[0];

            wgs.global = {output.get_shape()[0], 1, 1};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        };

        return f;
    }
};

class CTCLossImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::CTCLossImpl)

    Stage::Ptr ctc_loss = make_stage<CTCLossGenerator>();

    CTCLossImpl() : PrimitiveImplOCL(CTCLoss::get_type_info_static()) {}
    CTCLossImpl(const program_node& node, const kernel_impl_params& params) : CTCLossImpl() {
        add_stage(ctc_loss, params);
    }
    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<CTCLossImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> CTCLoss::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<ctc_loss>());
    return std::make_unique<CTCLossImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ctc_loss)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::CTCLossImpl)
