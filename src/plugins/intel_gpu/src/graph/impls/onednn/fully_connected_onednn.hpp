// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
#include "impls/onednn/utils.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <memory>
#include <cmath>

#define LOG_AND_RETURN_FALSE(node) do {                                         \
    GPU_DEBUG_TRACE << (node).id() << " :  Do not select onednn" << std::endl;  \
    return false;                                                               \
} while (0)

namespace cldnn {
namespace onednn {

struct FullyConnectedImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::fc")
    FullyConnectedImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            LOG_AND_RETURN_FALSE(node);

        const auto& fc_node = node.as<fully_connected>();
        const auto& in_layout = fc_node.get_input_layout(0);
        const auto& out_layout = fc_node.get_output_layout(0);
        auto in0_dt = in_layout.data_type;
        auto wei_dt = fc_node.weights().get_output_layout(false).data_type;
        auto out_dt = out_layout.data_type;
        auto fc_prim = fc_node.get_primitive();

        if (one_of(data_types::i64, {in0_dt, wei_dt}))
            LOG_AND_RETURN_FALSE(node);

        if (!everyone_is(format::bfyx, in_layout.format, out_layout.format) &&
            !everyone_is(format::bfzyx, in_layout.format, out_layout.format) &&
            !everyone_is(format::bfwzyx, in_layout.format, out_layout.format) &&
            !everyone_is(format::any, in_layout.format, out_layout.format))
            LOG_AND_RETURN_FALSE(node);

        if (!is_supported_pad(in_layout) || !is_supported_pad(out_layout))
            LOG_AND_RETURN_FALSE(node);

        bool f16f16_case = everyone_is(data_types::f16, in0_dt, wei_dt) && one_of(out_dt, {data_types::f16, data_types::f32, data_types::i8});
        bool f32f32_case = everyone_is(data_types::f32, in0_dt, wei_dt);
        bool u8s8_case = one_of(in0_dt, {data_types::i8, data_types::u8}) &&
                         one_of(wei_dt, {data_types::i8, data_types::u8}) &&
                         one_of(out_dt, {data_types::f16, data_types::f32, data_types::i32, data_types::i8, data_types::u8});
        bool compressed_case = fc_prim->compressed_weights &&
                               one_of(in0_dt, {data_types::f16, data_types::f32, data_types::i8, data_types::u8}) &&
                               one_of(wei_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}) &&
                               one_of(out_dt, {data_types::f16, data_types::f32, data_types::u8, data_types::i8});
        if (!f16f16_case && !f32f32_case && !u8s8_case && !compressed_case)
            LOG_AND_RETURN_FALSE(node);

        if (fc_prim->compressed_weights) {
            if (fc_prim->decompression_zero_point.is_valid()) {
                auto decompression_zp_idx = fc_prim->bias.is_valid() ? 4 : 3;
                auto decompression_zp_dt = fc_node.get_input_layout(decompression_zp_idx).data_type;
                if ((wei_dt != ov::element::Type_t::u4 && wei_dt != ov::element::Type_t::u8) ||
                    (decompression_zp_dt != ov::element::Type_t::u8 && decompression_zp_dt != ov::element::Type_t::i8)) {
                    LOG_AND_RETURN_FALSE(node);
                }
            }
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;
        }
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
