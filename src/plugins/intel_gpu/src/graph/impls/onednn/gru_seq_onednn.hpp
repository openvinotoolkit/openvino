// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_seq_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"
#include "transformations/utils/utils.hpp"
#include "impls/onednn/utils.hpp"

#include <memory>


namespace cldnn {
namespace onednn {

struct GRUSeqImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gru_seq")
    GRUSeqImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gru_seq>());
        return true;
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (info.arch == gpu_arch::unknown || !config.get_use_onednn())
            return false;
        const auto& gru_seq_node = node.as<gru_seq>();
        const auto& in_layout = gru_seq_node.get_input_layout(0);
        const auto& out_layout = gru_seq_node.get_output_layout(0);

        if (node.get_input_layout(0).format != cldnn::format::bfyx && node.get_input_layout(0).format != cldnn::format::fbyx
            && node.get_input_layout(0).format != cldnn::format::ybfx)
            return false;

        if (!is_supported_pad(in_layout) || !is_supported_pad(out_layout))
            return false;

        auto in0_dt = node.get_input_layout(0).data_type;
        auto in1_dt = node.get_input_layout(1).data_type;
        auto in2_dt = node.get_input_layout(2).data_type;
        auto in3_dt = node.get_input_layout(3).data_type;
        auto in4_dt = node.get_input_layout(4).data_type;
        auto out0_dt = node.get_output_layout(0).data_type;
        auto out1_dt = node.get_output_layout(1).data_type;
        bool f16_case = everyone_is(data_types::f16, in0_dt, in1_dt, in2_dt, in3_dt, in4_dt, out0_dt, out1_dt);
        bool f32_case = everyone_is(data_types::f32, in0_dt, in1_dt, in2_dt, in3_dt, in4_dt, out0_dt, out1_dt);
        bool u8u8u8_case = one_of(out0_dt, {data_types::u8, data_types::f32}) && everyone_is(data_types::i8, in3_dt) &&
            everyone_is(data_types::u8, in0_dt, in1_dt, out1_dt) && everyone_is(data_types::f32, in2_dt, in4_dt);
        bool f32u8f32_case = everyone_is(data_types::u8, in0_dt) && everyone_is(data_types::i8, in2_dt, in3_dt) &&
            one_of(out0_dt, {data_types::u8, data_types::f32}) && everyone_is(data_types::f32, in1_dt, in4_dt, out1_dt);
        bool s8s8s8_case = everyone_is(data_types::i8, in0_dt, in1_dt, out0_dt, out1_dt) && one_of(out0_dt, {data_types::i8, data_types::f32}) &&
            everyone_is(data_types::f32, in2_dt, in4_dt);
        bool f32s8f32_case = everyone_is(data_types::i8, in0_dt, in1_dt, in2_dt) && one_of(out0_dt, {data_types::i8, data_types::f32}) &&
            everyone_is(data_types::f32, in1_dt, in4_dt, out1_dt);

        return f16_case || f32_case || u8u8u8_case || f32u8f32_case || s8s8s8_case || f32s8f32_case;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<gru_seq>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0 ; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::get_default_format(out_rank);
        }
        out_fmts[0] = format::ybfx;
        out_fmts[1] = format::bfyx;

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
