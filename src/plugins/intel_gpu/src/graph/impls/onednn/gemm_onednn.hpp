// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "impls/registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct GemmImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gemm")
    GemmImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gemm>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;

        const auto& gemm_node = node.as<gemm>();
        const auto& gemm_prim = gemm_node.get_primitive();
        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& out_layout = node.get_output_layout(0);

        auto in0_dt = in0_layout.data_type;
        auto in1_dt = in1_layout.data_type;
        auto out_dt = out_layout.data_type;

        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
            format::bfxy,
            format::byxf,
            format::byfx,
            format::bxfy,
            format::fybx,  //format used for gemm fusion
            format::fyxb,  //format used for gemm fusion
            format::xbfy, // format used for gemm fusion
            format::ybfx, // format used for gemm fusion
            format::bfzyx,
            format::bfwzyx,
        };

        if (gemm_prim->alpha != 1.0f || gemm_prim->beta != 0.0f)
            return false;

        if (out_layout.data_padding)
            return false;

        if (one_of(in0_dt, {data_types::f32, data_types::i64}) || one_of(in1_dt, {data_types::f32, data_types::i64}))
            return false;

        if (!one_of(in0_layout.format.value, supported_formats) ||
            !one_of(in1_layout.format.value, supported_formats) ||
            !one_of(out_layout.format.value, supported_formats))
            return false;

        bool f16f16_case = everyone_is(data_types::f16, in0_dt, in1_dt) && one_of(out_dt, {data_types::f16, data_types::f32, data_types::i8});
        bool u8s8_case = one_of(in0_dt, {data_types::i8, data_types::u8}) &&
                         one_of(in1_dt, {data_types::i8, data_types::u8}) &&
                         one_of(out_dt, {data_types::f16, data_types::f32, data_types::i32, data_types::i8, data_types::u8});

        if (!f16f16_case && !u8s8_case)
            return false;

        if (gemm_prim->indirect_a || gemm_prim->indirect_b)
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<gemm>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            size_t out_rank = node.get_output_layout().get_rank();
            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;

            if (out_fmts[0] == format::any) {
                out_fmts[0] = target_format;
            }
        }

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
