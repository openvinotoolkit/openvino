// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xetla_lora.hpp"

#include "common_utils/kernel_generator_base.hpp"
#include "impls/ocl_v2/utils/jitter.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"
#include "xetla_postops.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_lora_build_options() {
    return " -Qxcm_jit_option=-DPASTokenReduction "
           " -mllvm --vc-disable-indvars-opt=true "
           " /Qxcm_jit_option=-enableBCR /Qxcm_doubleGRF "
           " -DXETLA_CODE_BASE=__CM__ ";
}

class XeTLALoraBaseGenerator : public KernelGenerator {
public:
    XeTLALoraBaseGenerator(std::string_view name, std::string_view suffix = "") : KernelGenerator(name, suffix) {}
    virtual ~XeTLALoraBaseGenerator() = default;
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_lora_build_options();
    }

protected:
    enum class MemLayout { row_major, col_major };

    struct Layouts {
        static constexpr MemLayout mem_layout_a = MemLayout::row_major;
        static constexpr MemLayout mem_layout_state_a = MemLayout::col_major;
        static constexpr MemLayout mem_layout_state_b = MemLayout::row_major;
        static constexpr MemLayout mem_layout_c = MemLayout::row_major;
    };

    std::string get_xetla_mem_layout(MemLayout layout) const {
        switch (layout) {
        case MemLayout::row_major:
            return "mem_layout::row_major";
        case MemLayout::col_major:
            return "mem_layout::col_major";
        default:
            assert(false, "Unsupported memory layout!");
        }
    }

    std::string ov_to_xetla_dtype(ov::element::Type type) const {
        switch (type) {
        case ov::element::Type_t::f16:
            return "fp16";
        case ov::element::Type_t::bf16:
            return "bf16";
        case ov::element::Type_t::f32:
            return "float";
        default:
            assert(false, "Unsupported data type!");
        }
    }

    void add_xetla_postops(const RuntimeParams& params,
                           std::vector<std::unique_ptr<XeTLAPostOP>>& xetla_postops,
                           size_t& post_op_index,
                           size_t& post_op_arg_index) const {
        for (const auto& postop : params.fused_desc) {
            const bool is_eltwise = fused_ops_are_one_of<eltwise>({postop});
            const bool is_activation = fused_ops_are_one_of<activation>({postop});
            if (is_eltwise) {
                auto eltwise = std::static_pointer_cast<const cldnn::eltwise>(postop.desc);
                auto eltwise_layout = params.input_layouts[post_op_arg_index++];
                auto eltwise_dtype = ov_to_xetla_dtype(eltwise_layout.data_type);

                bool broadcast = false;
                if (params.is_dynamic()) {
                    broadcast = !eltwise_layout.get_partial_shape()[0].is_dynamic() || !eltwise_layout.get_partial_shape()[1].is_dynamic();
                } else {
                    const auto eltwise_M = extract_channel(ChannelName::BATCH, eltwise_layout) * extract_channel(ChannelName::FEATURE, eltwise_layout);
                    broadcast = eltwise_M == 1;
                }

                const bool broadcast_start_0 = eltwise->broadcast_spec.m_axis == 0;
                assert(broadcast == broadcast_start_0);

                if (broadcast && broadcast_start_0) {
                    if (eltwise->mode == eltwise_mode::sum) {
                        xetla_postops.push_back(std::make_unique<ShiftChannels>(post_op_index++, eltwise_dtype));
                    } else if (eltwise->mode == eltwise_mode::prod) {
                        xetla_postops.push_back(std::make_unique<ScaleChannels>(post_op_index++, eltwise_dtype));
                    }
                } else {
                    const auto eltwise_op = get_xetla_eltwise_op(eltwise->mode);
                    assert(eltwise_op != Eltwise::EltwiseOp::none);
                    xetla_postops.push_back(std::make_unique<Eltwise>(post_op_index++, eltwise_dtype, eltwise_op));
                }
            } else if (is_activation) {
                const auto activation = std::static_pointer_cast<const cldnn::activation>(postop.desc);
                const auto activation_dtype = ov_to_xetla_dtype(ov::element::Type_t::f32);
                const auto activation_op = get_xetla_activation_op(activation->activation_function);

                assert(activation_op != Activation::ActivationOp::none);
                xetla_postops.push_back(std::make_unique<Activation>(post_op_index++, activation_dtype, activation_op));
            }
        }
    }

    using range_t = std::vector<uint64_t>;
    static range_t get_group_range(size_t matrix_m, size_t matrix_n, size_t wg_tile_m, size_t wg_tile_n, size_t num_global_kslicing) {
        size_t group_range_m = (matrix_m + wg_tile_m - 1u) / wg_tile_m;
        size_t group_range_n = (matrix_n + wg_tile_n - 1u) / wg_tile_n;
        return {num_global_kslicing, group_range_m, group_range_n};
    }

    static range_t get_local_range(size_t wg_tile_m, size_t wg_tile_n, size_t sg_tile_m, size_t sg_tile_n, size_t num_local_kslicing) {
        size_t local_range_m = (wg_tile_m + sg_tile_m - 1u) / sg_tile_m;
        size_t local_range_n = (wg_tile_n + sg_tile_n - 1u) / sg_tile_n;
        assert(local_range_m * local_range_n * num_local_kslicing <= 32);
        return {num_local_kslicing, local_range_m, local_range_n};
    }

    static std::tuple<range_t, range_t> get_nd_range(size_t matrix_m,
                                                     size_t matrix_n,
                                                     size_t wg_tile_m,
                                                     size_t wg_tile_n,
                                                     size_t sg_tile_m,
                                                     size_t sg_tile_n,
                                                     size_t num_global_kslicing,
                                                     size_t num_local_kslicing) {
        auto local_range = get_local_range(wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, num_local_kslicing);
        auto group_range = get_group_range(matrix_m, matrix_n, wg_tile_m, wg_tile_n, num_global_kslicing);
        return {local_range, group_range};
    }

    static bool is_2dload_aligned(size_t size, ov::element::Type dtype) {
        static constexpr size_t min_size = 64 * 8;
        static constexpr size_t multiple_of = 16 * 8;
        auto dtype_size = ov::element::Type(dtype).bitwidth();
        auto size_in_bits = size * dtype_size;
        return size_in_bits >= min_size && size_in_bits % multiple_of == 0;
    }

    struct Tiling {
        const size_t wg_m;
        const size_t wg_n;
        const size_t sg_m;
        const size_t sg_n;
        const size_t sg_k;
        const size_t num_global_kslicing;
        const size_t num_local_kslicing;

        Tiling(const RuntimeParams& params) : wg_m{256}, wg_n{128}, sg_m{32}, sg_n{32}, sg_k{32}, num_global_kslicing{1}, num_local_kslicing{1} {}
    };

public:
    struct LoraShapeUtils {
        std::tuple<size_t, size_t, size_t, size_t> get_lora_gemm_shape(const RuntimeParams& params) const {
            return {LoraShapeUtils::get_total_tokens(params),
                    LoraShapeUtils::get_lora_rank(params),
                    LoraShapeUtils::get_hidden_size_input(params),
                    LoraShapeUtils::get_hidden_size_output(params)};
        }

        static size_t get_total_tokens(const RuntimeParams& params) {
            assert(!params.is_dynamic());
            return extract_channel(ChannelName::BATCH, params.output_layouts[0]) * extract_channel(ChannelName::FEATURE, params.output_layouts[0]);
        }

        static size_t get_hidden_size_input(const RuntimeParams& params) {
            assert(!params.is_dynamic());
            return extract_channel(ChannelName::Y, params.input_layouts[1]) * extract_channel(ChannelName::X, params.input_layouts[1]);
        }

        static size_t get_hidden_size_output(const RuntimeParams& params) {
            assert(!params.is_dynamic());
            return extract_channel(ChannelName::Y, params.output_layouts[0]) * extract_channel(ChannelName::X, params.output_layouts[0]);
        }

        static size_t get_lora_rank(const RuntimeParams& params) {
            assert(!params.is_dynamic());
            return extract_channel(ChannelName::FEATURE, params.input_layouts[3]);
        }

        static auto get_total_tokens_jit(const RuntimeParams& params) {
            ov::intel_gpu::ocl::LayoutJitter jit(params.input_layouts[1], params.in_port_to_shape_info_offset.at(1));
            const auto jit_val = "(" + jit.dim(ChannelName::BATCH) + " * " + jit.dim(ChannelName::FEATURE) + ")";
            return jit_val;
        }
        static auto get_lora_rank_jit(const RuntimeParams& params) {
            ov::intel_gpu::ocl::LayoutJitter jit(params.input_layouts[3], params.in_port_to_shape_info_offset.at(3));
            const auto jit_val = jit.dim(ChannelName::FEATURE);
            return jit_val;
        }

        static auto get_hidden_size_input_jit(const RuntimeParams& params) {
            ov::intel_gpu::ocl::LayoutJitter jit(params.input_layouts[1], params.in_port_to_shape_info_offset.at(1));
            const auto jit_val = "(" + jit.dim(ChannelName::Y) + " * " + jit.dim(ChannelName::X) + ")";
            return jit_val;
        }
        static auto get_hidden_size_output_jit(const RuntimeParams& params) {
            ov::intel_gpu::ocl::LayoutJitter jit(params.output_layouts[0], params.out_port_to_shape_info_offset.at(0));
            const auto jit_val = "(" + jit.dim(ChannelName::Y) + " * " + jit.dim(ChannelName::X) + ")";
            return jit_val;
        }
    };
};

class XetlaLoRAFusedGenerator : public XeTLALoraBaseGenerator {
public:
    XetlaLoRAFusedGenerator() : XeTLALoraBaseGenerator("xetla_lora_fused", "") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto mem_layout_a = Layouts::mem_layout_a;
        const auto mem_layout_state_a = Layouts::mem_layout_state_a;
        const auto mem_layout_state_b = Layouts::mem_layout_state_b;
        const auto mem_layout_c = Layouts::mem_layout_c;

        const uint32_t temp_in_reg = 1;

        uint32_t fused_wg_m = 8;
        uint32_t fused_sg_m = 8;

        uint32_t fusedA_wg_n = 64;
        uint32_t fusedA_sg_n = 64;
        uint32_t fusedA_sg_k = 32;
        uint32_t fused_local_kslicing = 1;

        uint32_t fusedB_total_wg_n = 256;
        uint32_t fusedB_wg_n = 128;
        uint32_t fusedB_sg_n = 32;
        uint32_t fusedB_sg_k = 32;

        jit.add({make_jit_constant("KERNEL_NAME", get_entry_point(params)),
                 make_jit_constant("LORA_DTYPE_A", ov_to_xetla_dtype(params.input_layouts[1].data_type)),
                 make_jit_constant("LORA_DTYPE_B", ov_to_xetla_dtype(params.input_layouts[2].data_type)),
                 make_jit_constant("LORA_DTYPE_C", ov_to_xetla_dtype(params.output_layouts[0].data_type)),
                 make_jit_constant("LORA_DTYPE_ACC", ov_to_xetla_dtype(ov::element::Type_t::f32)),
                 make_jit_constant("LORA_SIZE_RANK", LoraShapeUtils::get_lora_rank_jit(params)),
                 make_jit_constant("LORA_WG_M", fused_wg_m),
                 make_jit_constant("LORA_WG_N_A", fusedA_wg_n),
                 make_jit_constant("LORA_WG_N_B", fusedB_wg_n),
                 make_jit_constant("LORA_SG_M", fused_sg_m),
                 make_jit_constant("LORA_SG_N_A", fusedA_sg_n),
                 make_jit_constant("LORA_SG_N_B", fusedB_sg_n),
                 make_jit_constant("LORA_SG_K_A", fusedA_sg_k),
                 make_jit_constant("LORA_SG_K_B", fusedB_sg_k),
                 make_jit_constant("LORA_WG_B_TOTAL", fusedB_total_wg_n),
                 make_jit_constant("LORA_LOCAL_SLICING", fused_local_kslicing),
                 make_jit_constant("LORA_MMA_ENGINE", "mma_engine::xmx"),
                 make_jit_constant("LORA_MEM_LAYOUT_A", get_xetla_mem_layout(mem_layout_a)),
                 make_jit_constant("LORA_MEM_LAYOUT_STATE_A", get_xetla_mem_layout(mem_layout_state_a)),
                 make_jit_constant("LORA_MEM_LAYOUT_STATE_B", get_xetla_mem_layout(mem_layout_state_b)),
                 make_jit_constant("LORA_MEM_LAYOUT_C", get_xetla_mem_layout(mem_layout_c)),
                 make_jit_constant("LORA_MEM_SPACE_TEMP", "mem_space::global"),
                 make_jit_constant("LORA_UNALIGNED", "false"),
                 make_jit_constant("LORA_TEMP_IN_REG", temp_in_reg),
                 make_jit_constant("LORA_SIZE_M", LoraShapeUtils::get_total_tokens_jit(params)),
                 make_jit_constant("LORA_SIZE_K", LoraShapeUtils::get_hidden_size_input_jit(params)),
                 make_jit_constant("LORA_SIZE_N", LoraShapeUtils::get_hidden_size_output_jit(params))});

        if (params.is_dynamic()) {
            jit.add({make_jit_constant("XETLA_SHAPE_INFO_ARG", "int *shape_info [[type(\"svmptr_t\")]],")});
        } else {
            jit.add({make_jit_constant("XETLA_SHAPE_INFO_ARG", "")});
        }

        size_t post_op_index = 0;
        size_t post_op_arg_index = 5;
        std::vector<std::unique_ptr<XeTLAPostOP>> xetla_postops;
        xetla_postops.push_back(std::make_unique<Eltwise>(post_op_index++, ov_to_xetla_dtype(params.input_layouts[0].data_type), Eltwise::EltwiseOp::sum));
        add_xetla_postops(params, xetla_postops, post_op_index, post_op_arg_index);

        auto post_op_definitions = generate_post_ops(xetla_postops);
        for (const auto& [name, value] : post_op_definitions) {
            jit.add({make_jit_constant(name, value)});
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});            // lora_input
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});            // state_a
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});            // state_alpha
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});            // state_b
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});           // out
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // temp
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});            // main_input

        if (params.has_fused_primitives()) {
            size_t num_fused_deps = 0;
            for (const auto& fd : params.fused_desc) {
                for (const auto& in_d : fd.inputs) {
                    if (in_d.m_type == cldnn::FusedInputType::EXTERNAL) {
                        num_fused_deps++;
                    }
                }
            }
            for (size_t i = 0; i < num_fused_deps; i++) {
                args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, static_cast<uint32_t>(i)});
            }
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const Tiling tiling{params};

            uint32_t fused_wg_m = 8;
            uint32_t fused_sg_m = 8;

            uint32_t fusedA_wg_n = 64;
            uint32_t fusedA_sg_n = 64;
            uint32_t fusedA_sg_k = 32;
            uint32_t fused_local_kslicing = 1;

            uint32_t fusedB_total_wg_n = 256;
            uint32_t fusedB_wg_n = 128;
            uint32_t fusedB_sg_n = 32;
            uint32_t fusedB_sg_k = 32;

            uint32_t local_range_m = (fused_wg_m + fused_sg_m - 1) / fused_sg_m;
            uint32_t local_range_nA = (fusedA_wg_n + fusedA_sg_n - 1) / fusedA_sg_n;
            uint32_t local_range_nB = (fusedB_wg_n + fusedB_sg_n - 1) / fusedB_sg_n;
            uint32_t local_range_n = local_range_nA > local_range_nB ? local_range_nA : local_range_nB;

            assert(local_range_m * local_range_n * fused_local_kslicing <= 32);
            uint32_t group_range_m = (LoraShapeUtils::get_total_tokens(params) + fused_wg_m - 1) / fused_wg_m;
            uint32_t group_range_n = (LoraShapeUtils::get_hidden_size_output(params) + fusedB_total_wg_n - 1) / fusedB_total_wg_n;

            auto [subgroup_range, group_range] = get_nd_range(LoraShapeUtils::get_total_tokens(params),
                                                              LoraShapeUtils::get_lora_rank(params),
                                                              tiling.wg_m,
                                                              tiling.wg_n,
                                                              tiling.sg_m,
                                                              tiling.sg_n,
                                                              tiling.num_global_kslicing,
                                                              tiling.num_local_kslicing);
            wgs.global = {group_range_n * local_range_n, group_range_m * local_range_m, 1};
            wgs.local = {local_range_n, local_range_m, fused_local_kslicing};
        }};
    }
};

class XetlaLoRAGEMMAGenerator : public XeTLALoraBaseGenerator {
    const bool is_aligned;

public:
    XetlaLoRAGEMMAGenerator(bool is_aligned) : XeTLALoraBaseGenerator("xetla_lora_gemmA", "A"), is_aligned{is_aligned} {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto mem_layout_a = Layouts::mem_layout_a;
        const auto mem_layout_b = Layouts::mem_layout_state_a;
        const auto mem_layout_c = Layouts::mem_layout_c;

        const Tiling tiling{params};
        jit.add({make_jit_constant("KERNEL_NAME", get_entry_point(params)),
                 make_jit_constant("LORA_DTYPE_A", ov_to_xetla_dtype(params.input_layouts[1].data_type)),
                 make_jit_constant("LORA_DTYPE_B", ov_to_xetla_dtype(params.input_layouts[2].data_type)),
                 make_jit_constant("LORA_DTYPE_C", ov_to_xetla_dtype(params.output_layouts[0].data_type)),
                 make_jit_constant("LORA_DTYPE_ACC", ov_to_xetla_dtype(ov::element::Type_t::f32)),
                 make_jit_constant("LORA_WG_M", tiling.wg_m),
                 make_jit_constant("LORA_WG_N", tiling.wg_n),
                 make_jit_constant("LORA_SG_M", tiling.sg_m),
                 make_jit_constant("LORA_SG_N", tiling.sg_n),
                 make_jit_constant("LORA_SG_K", tiling.sg_k),
                 make_jit_constant("LORA_GLOBAL_SLICING", tiling.num_global_kslicing),
                 make_jit_constant("LORA_LOCAL_SLICING", tiling.num_local_kslicing),
                 make_jit_constant("LORA_MMA_ENGINE", "mma_engine::xmx"),
                 make_jit_constant("LORA_MEM_LAYOUT_A", get_xetla_mem_layout(mem_layout_a)),
                 make_jit_constant("LORA_MEM_LAYOUT_B", get_xetla_mem_layout(mem_layout_b)),
                 make_jit_constant("LORA_MEM_LAYOUT_C", get_xetla_mem_layout(mem_layout_c)),
                 make_jit_constant("LORA_SIZE_M", LoraShapeUtils::get_total_tokens_jit(params)),
                 make_jit_constant("LORA_SIZE_K", LoraShapeUtils::get_hidden_size_input_jit(params)),
                 make_jit_constant("LORA_SIZE_N", LoraShapeUtils::get_lora_rank_jit(params)),
                 make_jit_constant("LORA_UNALIGNED", !is_aligned)});

        if (params.is_dynamic()) {
            jit.add({make_jit_constant("XETLA_SHAPE_INFO_ARG", "int *shape_info [[type(\"svmptr_t\")]],")});
        } else {
            jit.add({make_jit_constant("XETLA_SHAPE_INFO_ARG", "")});
        }

        std::vector<std::unique_ptr<XeTLAPostOP>> xetla_postops;
        xetla_postops.push_back(std::make_unique<ScaleChannels>(0, ov_to_xetla_dtype(params.input_layouts[2].data_type)));

        auto post_op_definitions = generate_post_ops(xetla_postops);
        for (const auto& [name, value] : post_op_definitions) {
            jit.add({make_jit_constant(name, value)});
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});            // lora_input
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});            // state_a
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});            // state_alpha
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // temp
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // acc
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // cnt

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const Tiling tiling{params};
            auto [subgroup_range, group_range] = get_nd_range(LoraShapeUtils::get_total_tokens(params),
                                                              LoraShapeUtils::get_lora_rank(params),
                                                              tiling.wg_m,
                                                              tiling.wg_n,
                                                              tiling.sg_m,
                                                              tiling.sg_n,
                                                              tiling.num_global_kslicing,
                                                              tiling.num_local_kslicing);
            wgs.global = {group_range[2] * subgroup_range[2], group_range[1] * subgroup_range[1], 1};
            wgs.local = {subgroup_range[2], subgroup_range[1], 1};
        }};
    }
};

class XetlaLoRAGEMMBGenerator : public XeTLALoraBaseGenerator {
    const bool is_aligned;

public:
    XetlaLoRAGEMMBGenerator(bool is_aligned) : XeTLALoraBaseGenerator("xetla_lora_gemmB", "B"), is_aligned{is_aligned} {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto mem_layout_a = Layouts::mem_layout_a;
        const auto mem_layout_b = Layouts::mem_layout_state_b;
        const auto mem_layout_c = Layouts::mem_layout_c;

        const Tiling tiling{params};
        jit.add({make_jit_constant("KERNEL_NAME", get_entry_point(params)),
                 make_jit_constant("LORA_DTYPE_A", ov_to_xetla_dtype(params.input_layouts[1].data_type)),
                 make_jit_constant("LORA_DTYPE_B", ov_to_xetla_dtype(params.input_layouts[4].data_type)),
                 make_jit_constant("LORA_DTYPE_C", ov_to_xetla_dtype(params.output_layouts[0].data_type)),
                 make_jit_constant("LORA_DTYPE_ACC", ov_to_xetla_dtype(ov::element::Type_t::f32)),
                 make_jit_constant("LORA_WG_M", tiling.wg_m),
                 make_jit_constant("LORA_WG_N", tiling.wg_n),
                 make_jit_constant("LORA_SG_M", tiling.sg_m),
                 make_jit_constant("LORA_SG_N", tiling.sg_n),
                 make_jit_constant("LORA_SG_K", tiling.sg_k),
                 make_jit_constant("LORA_GLOBAL_SLICING", tiling.num_global_kslicing),
                 make_jit_constant("LORA_LOCAL_SLICING", tiling.num_local_kslicing),
                 make_jit_constant("LORA_MMA_ENGINE", "mma_engine::xmx"),
                 make_jit_constant("LORA_MEM_LAYOUT_A", get_xetla_mem_layout(mem_layout_a)),
                 make_jit_constant("LORA_MEM_LAYOUT_B", get_xetla_mem_layout(mem_layout_b)),
                 make_jit_constant("LORA_MEM_LAYOUT_C", get_xetla_mem_layout(mem_layout_c)),
                 make_jit_constant("LORA_SIZE_M", LoraShapeUtils::get_total_tokens_jit(params)),
                 make_jit_constant("LORA_SIZE_K", LoraShapeUtils::get_lora_rank_jit(params)),
                 make_jit_constant("LORA_SIZE_N", LoraShapeUtils::get_hidden_size_output_jit(params)),
                 make_jit_constant("LORA_UNALIGNED", !is_aligned)});

        if (params.is_dynamic()) {
            jit.add({make_jit_constant("XETLA_SHAPE_INFO_ARG", "int *shape_info [[type(\"svmptr_t\")]],")});
        } else {
            jit.add({make_jit_constant("XETLA_SHAPE_INFO_ARG", "")});
        }

        size_t post_op_index = 0;
        size_t post_op_arg_index = 5;
        std::vector<std::unique_ptr<XeTLAPostOP>> xetla_postops;
        xetla_postops.push_back(std::make_unique<Eltwise>(post_op_index++, ov_to_xetla_dtype(params.input_layouts[0].data_type), Eltwise::EltwiseOp::sum));
        add_xetla_postops(params, xetla_postops, post_op_index, post_op_arg_index);

        auto post_op_definitions = generate_post_ops(xetla_postops);
        for (const auto& [name, value] : post_op_definitions) {
            jit.add({make_jit_constant(name, value)});
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // temp
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});            // state_b
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});           // out
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // acc
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // cnt
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});            // main_input

        if (params.has_fused_primitives()) {
            size_t num_fused_deps = 0;
            for (const auto& fd : params.fused_desc) {
                for (const auto& in_d : fd.inputs) {
                    if (in_d.m_type == cldnn::FusedInputType::EXTERNAL) {
                        num_fused_deps++;
                    }
                }
            }
            for (size_t i = 0; i < num_fused_deps; i++) {
                args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, static_cast<uint32_t>(i)});
            }
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const Tiling tiling{params};
            auto [subgroup_range, group_range] = get_nd_range(LoraShapeUtils::get_total_tokens(params),
                                                              LoraShapeUtils::get_hidden_size_output(params),
                                                              tiling.wg_m,
                                                              tiling.wg_n,
                                                              tiling.sg_m,
                                                              tiling.sg_n,
                                                              tiling.num_global_kslicing,
                                                              tiling.num_local_kslicing);
            wgs.global = {group_range[2] * subgroup_range[2], group_range[1] * subgroup_range[1], 1};
            wgs.local = {subgroup_range[2], subgroup_range[1], 1};
        }};
    }
};

class LoRAImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::LoRAImpl)
    Stage::Ptr lora_fused = make_stage<XetlaLoRAFusedGenerator>();
    // Stage::Ptr lora_gemm_a = make_stage<XetlaLoRAGEMMAGenerator>(true);
    // Stage::Ptr lora_gemm_a_unaligned = make_stage<XetlaLoRAGEMMAGenerator>(false);

    // Stage::Ptr lora_gemm_b = make_stage<XetlaLoRAGEMMBGenerator>(true);
    // Stage::Ptr lora_gemm_b_unaligned = make_stage<XetlaLoRAGEMMBGenerator>(false);

    LoRAImpl() : PrimitiveImplOCL(LoRAImplementationManager::get_type_info_static()) {}
    LoRAImpl(const program_node& node, const RuntimeParams& params) : LoRAImpl() {
        add_stage(lora_fused, params);
        // add_stage(lora_gemm_a, params);
        // add_stage(lora_gemm_b, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<LoRAImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        size_t buf_size = XeTLALoraBaseGenerator::LoraShapeUtils::get_total_tokens(params) * XeTLALoraBaseGenerator::LoraShapeUtils::get_lora_rank(params);
        return {BufferDescriptor{buf_size, ov::element::f16}, BufferDescriptor{0, ov::element::f32}, BufferDescriptor{0, ov::element::u32}};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> LoRAImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<lora>());
    return std::make_unique<LoRAImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::LoRAImpl)
