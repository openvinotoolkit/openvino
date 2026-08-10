// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct GemmImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gemm")
    GemmImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gemm>());
        const auto& config = node.get_program().get_config();
        if (!config.get_use_onednn())
            return false;
#ifdef OV_GPU_VENDOR_INTEL
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;
#endif

        const auto& gemm_node = node.as<gemm>();
        const auto& gemm_prim = gemm_node.get_primitive();
        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& out_layout = node.get_output_layout(0);

        // oneDNN GEMM requires at least 2D tensors with a known (static) rank
        // Check actual shape rank, not format rank
        auto has_rank_less_than = [](const ov::PartialShape& ps, int64_t min_rank) {
            const auto& r = ps.rank();
            return r.is_dynamic() || r.get_length() < min_rank;
        };

        const auto& in0_shape = in0_layout.get_partial_shape();
        const auto& in1_shape = in1_layout.get_partial_shape();
        const auto& out_shape = out_layout.get_partial_shape();
        if (has_rank_less_than(in0_shape, 2) ||
            has_rank_less_than(in1_shape, 2) ||
            has_rank_less_than(out_shape, 2))
            return false;

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

#ifdef OV_GPU_VENDOR_INTEL
        if (one_of(in0_dt, {data_types::f32, data_types::i64}) || one_of(in1_dt, {data_types::f32, data_types::i64}))
            return false;
#else
        if (one_of(in0_dt, {data_types::i64}) || one_of(in1_dt, {data_types::i64}))
            return false;
#endif

        if (!one_of(in0_layout.format.value, supported_formats) ||
            !one_of(in1_layout.format.value, supported_formats) ||
            !one_of(out_layout.format.value, supported_formats))
            return false;

#ifdef OV_GPU_VENDOR_NVIDIA
        // cuDNN matmul doesn't support 5D/6D format tags (abcde, abcdef)
        // Reject rank > 3 to fall back to OCL kernels
        if (in0_shape.rank().is_static() && in0_shape.rank().get_length() > 3)
            return false;
        if (in1_shape.rank().is_static() && in1_shape.rank().get_length() > 3)
            return false;
        if (out_shape.rank().is_static() && out_shape.rank().get_length() > 3)
            return false;

        // cuDNN matmul only supports a limited set of activation post-ops:
        // relu, tanh, elu, logistic (all with alpha=0), and at most one.
        {
            int activation_count = 0;
            for (auto& fo : gemm_node.get_fused_primitives()) {
                if (fo.is_type<activation>()) {
                    auto act_prim = fo.typed_desc<activation>();
                    auto func = act_prim->activation_function;
                    if (func == activation_func::none)
                        continue;
                    if (++activation_count > 1)
                        return false;
                    bool supported = (func == activation_func::relu ||
                                      func == activation_func::hyperbolic_tan ||
                                      func == activation_func::elu ||
                                      func == activation_func::logistic);
                    if (!supported)
                        return false;
                    if (act_prim->additional_params.a != 0.0f)
                        return false;
                } else {
                    return false;
                }
            }
        }
#endif

        //format bxfy not on whitelist for fusable_output_order_onednn in gemm_inst.h what makes assert error later
        //skipped onednn implementation as ocl is faster
        std::vector<size_t> out_transpose_order(std::begin(gemm_prim->output_transpose_order), std::end(gemm_prim->output_transpose_order));
        if (out_transpose_order == format::traits(format::bxfy)._order) {
            return false;
        }

        bool f16f16_case = everyone_is(data_types::f16, in0_dt, in1_dt) && one_of(out_dt, {data_types::f16, data_types::f32, data_types::i8});
        bool u8s8_case = one_of(in0_dt, {data_types::i8, data_types::u8}) &&
                         one_of(in1_dt, {data_types::i8, data_types::u8}) &&
                         one_of(out_dt, {data_types::f16, data_types::f32, data_types::i32, data_types::i8, data_types::u8});

#ifdef OV_GPU_VENDOR_INTEL
        if (!f16f16_case && !u8s8_case)
            return false;
#else
        bool f32f32_case = everyone_is(data_types::f32, in0_dt, in1_dt) && one_of(out_dt, {data_types::f32});
        if (!f32f32_case && !f16f16_case && !u8s8_case)
            return false;
#endif

        if (gemm_prim->indirect_a || gemm_prim->indirect_b)
            return false;

        // Validate transpose patterns are fusable with oneDNN
        auto expand_transpose_order = [](const std::vector<int64_t>& transpose_order, size_t rank) -> std::vector<size_t> {
            if (transpose_order.size() == rank) {
                std::vector<size_t> order;
                order.reserve(rank);
                for (auto v : transpose_order) {
                    if (v < 0 || static_cast<size_t>(v) >= rank)
                        return {};
                    order.push_back(static_cast<size_t>(v));
                }
                return order;
            }
            if (transpose_order.size() > rank) {
                return {};
            }
            const size_t add = rank - transpose_order.size();
            std::vector<size_t> order;
            order.reserve(rank);
            for (size_t i = 0; i < add; ++i)
                order.push_back(i);
            for (auto v : transpose_order) {
                if (v < 0 || static_cast<size_t>(v) >= (rank - add))
                    return {};
                order.push_back(static_cast<size_t>(v) + add);
            }

            return order;
        };

        auto has_non_identity_transpose = [](const std::vector<int64_t>& order) {
            for (size_t i = 0; i < order.size(); ++i) {
                if (static_cast<size_t>(order[i]) != i)
                    return true;
            }
            return false;
        };

        auto is_xy_only_transpose = [](const std::vector<size_t>& order) {
            if (order.size() < 2)
                return false;

            for (size_t i = 0; i + 2 < order.size(); ++i) {
                if (order[i] != i)
                    return false;
            }

            const size_t last = order.size() - 1;
            return order[last - 1] == last && order[last] == last - 1;
        };

        // Helper to get actual tensor rank from PartialShape, fallback to format rank if dynamic
        auto get_transpose_rank = [](const ov::PartialShape& shape, size_t format_rank) -> size_t {
            auto rank = shape.rank();
            if (rank.is_static()) {
                return static_cast<size_t>(rank.get_length());
            }
            return format_rank;  // Fallback to format rank only when shape rank is dynamic
        };

        auto validate_input_transpose = [&](const std::vector<int64_t>& transpose_order, size_t rank) {
            auto order = expand_transpose_order(transpose_order, rank);
            if (order.empty())
                return false;
            if (is_xy_only_transpose(order))
                return true;
            if (order.size() == 4) {
                format dummy_fmt(format::bfyx);
                return gemm_inst::is_fusable_permute_input_order_onednn(order, dummy_fmt);
            }
            return false;
        };

        auto validate_output_transpose = [&](const std::vector<int64_t>& transpose_order, size_t rank) {
            auto order = expand_transpose_order(transpose_order, rank);
            if (order.empty())
                return false;
            if (is_xy_only_transpose(order))
                return true;
            if (order.size() == 4) {
                format dummy_fmt(format::bfyx);
                return gemm_inst::is_fusable_permute_output_order_onednn(order, dummy_fmt);
            }
            return false;
        };

        // Check input0 transpose (use PartialShape rank, not format rank)
        if (has_non_identity_transpose(gemm_prim->input0_transpose_order)) {
            size_t in0_rank = get_transpose_rank(in0_shape, in0_layout.get_rank());
            if (!validate_input_transpose(gemm_prim->input0_transpose_order, in0_rank))
                return false;
        }

        // Check input1 transpose (use PartialShape rank, not format rank)
        if (has_non_identity_transpose(gemm_prim->input1_transpose_order)) {
            size_t in1_rank = get_transpose_rank(in1_shape, in1_layout.get_rank());
            if (!validate_input_transpose(gemm_prim->input1_transpose_order, in1_rank))
                return false;
        }

        // Check output transpose (use PartialShape rank, not format rank)
        if (has_non_identity_transpose(gemm_prim->output_transpose_order)) {
            size_t out_rank = get_transpose_rank(out_shape, out_layout.get_rank());
            if (!validate_output_transpose(gemm_prim->output_transpose_order, out_rank))
                return false;
        }

        // Validate matmul broadcast semantics early to avoid runtime oneDNN PD failures
        // on dynamic reshape/transpose patterns that cannot produce a valid dst shape.
        {
            std::vector<layout> transformed_in_layouts{in0_layout, in1_layout};
            if (gemm_prim->dependencies().size() == 3) {
                transformed_in_layouts.emplace_back(node.get_input_layout(2));
            }

            transformed_in_layouts = gemm_inst::transform_input_layouts(gemm_prim,
                                                                         transformed_in_layouts,
                                                                         node.get_program().is_new_shape_infer());
            auto transformed_out_layout = gemm_inst::transform_output_layout(gemm_prim,
                                                                              transformed_in_layouts,
                                                                              out_layout);

            auto apply_order_to_dims = [&](const ov::PartialShape& pshape,
                                           const std::vector<int64_t>& transpose_order,
                                           size_t format_rank) -> std::vector<ov::Dimension> {
                const size_t rank = get_transpose_rank(pshape, format_rank);
                auto order = expand_transpose_order(transpose_order, rank);
                if (order.empty())
                    return {};

                std::vector<ov::Dimension> dims(rank, ov::Dimension::dynamic());
                for (size_t i = 0; i < rank; ++i) {
                    if (pshape.rank().is_static() && order[i] < pshape.size()) {
                        dims[i] = pshape[order[i]];
                    }
                }
                return dims;
            };

            auto in0_dims = apply_order_to_dims(transformed_in_layouts[0].get_partial_shape(),
                                                gemm_prim->input0_transpose_order,
                                                transformed_in_layouts[0].get_rank());
            auto in1_dims = apply_order_to_dims(transformed_in_layouts[1].get_partial_shape(),
                                                gemm_prim->input1_transpose_order,
                                                transformed_in_layouts[1].get_rank());
            auto out_dims = apply_order_to_dims(transformed_out_layout.get_partial_shape(),
                                                gemm_prim->output_transpose_order,
                                                transformed_out_layout.get_rank());

            if (in0_dims.empty() || in1_dims.empty() || out_dims.empty())
                return false;

            if (in0_dims.size() != in1_dims.size() || in0_dims.size() != out_dims.size() || in0_dims.size() < 2)
                return false;

            const size_t rank = in0_dims.size();
            for (size_t i = 0; i + 2 < rank; ++i) {
                const auto& a = in0_dims[i];
                const auto& b = in1_dims[i];
                const auto& c = out_dims[i];

                if (a.is_static() && b.is_static()) {
                    const auto av = a.get_length();
                    const auto bv = b.get_length();
                    if (av != 1 && bv != 1 && av != bv)
                        return false;

                    if (c.is_static()) {
                        const auto expected = (av == 1) ? bv : ((bv == 1) ? av : av);
                        if (c.get_length() != expected)
                            return false;
                    }
                }

                if (c.is_static() && c.get_length() == 1) {
                    if ((a.is_static() && a.get_length() > 1) ||
                        (b.is_static() && b.get_length() > 1)) {
                        return false;
                    }
                }
            }

            const size_t m_idx = rank - 2;
            const size_t n_idx = rank - 1;
            if (out_dims[m_idx].is_static() && out_dims[m_idx].get_length() == 1 &&
                in0_dims[m_idx].is_static() && in0_dims[m_idx].get_length() > 1) {
                return false;
            }
            if (out_dims[n_idx].is_static() && out_dims[n_idx].get_length() == 1 &&
                in1_dims[n_idx].is_static() && in1_dims[n_idx].get_length() > 1) {
                return false;
            }
        }

        // When all batch dims are 1, oneDNN collapses to a 2D matmul descriptor.
        // Binary post-ops with higher rank are rejected by oneDNN at primitive creation.
        if (!in0_shape.is_dynamic() && !in1_shape.is_dynamic() && !out_shape.is_dynamic()) {
            std::vector<layout> tmp_layouts{in0_layout, in1_layout};
            if (gemm_prim->dependencies().size() == 3)
                tmp_layouts.emplace_back(node.get_input_layout(2));
            auto tmp_in = gemm_inst::transform_input_layouts(gemm_prim, tmp_layouts, node.get_program().is_new_shape_infer());
            auto tmp_out = gemm_inst::transform_output_layout(gemm_prim, tmp_in, out_layout);

            const auto& t0 = tmp_in[0];
            const auto& t1 = tmp_in[1];
            bool will_collapse = !t0.is_dynamic() && !t1.is_dynamic() && !tmp_out.is_dynamic()
                                 && t0.count() != 0 && t1.count() != 0 && tmp_out.count() != 0
                                 && t0.count() / (t0.spatial(0) * t0.spatial(1)) == 1
                                 && t1.count() / (t1.spatial(0) * t1.spatial(1)) == 1
                                 && tmp_out.count() / (tmp_out.spatial(0) * tmp_out.spatial(1)) == 1;

            if (will_collapse) {
                for (const auto& fp : gemm_node.get_fused_primitives()) {
                    if (!fp.is_type<eltwise>() || !fp.has_outer_dep())
                        continue;
                    const auto& dep_l = node.get_dependency(fp.outer_dep_start_idx).get_output_layout(false);
                    const auto dep_ps = dep_l.get_partial_shape();
                    if (dep_l.is_dynamic() || !dep_ps.rank().is_static() || dep_ps.rank().get_length() > 2)
                        return false;
                }
            }
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<gemm>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        auto target_format = format::get_default_format(out_rank);

        for (size_t idx = 0 ; idx < in_fmts.size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            in_fmts[idx] = target_format;
        }
        out_fmts[0] = target_format;

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
