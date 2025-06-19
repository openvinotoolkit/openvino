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
#include "utils/xetla_postops.hpp"

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

    enum class MemLayout { row_major, col_major };

    struct Layouts {
        static constexpr MemLayout mem_layout_a = MemLayout::row_major;
        static constexpr MemLayout mem_layout_state_a = MemLayout::col_major;
        static constexpr MemLayout mem_layout_state_b = MemLayout::row_major;
        static constexpr MemLayout mem_layout_temp = MemLayout::row_major;
        static constexpr MemLayout mem_layout_c = MemLayout::row_major;
    };

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
    };

protected:
    std::string get_xetla_mem_layout(MemLayout layout) const {
        switch (layout) {
        case MemLayout::row_major:
            return "mem_layout::row_major";
        case MemLayout::col_major:
            return "mem_layout::col_major";
        default:
            OPENVINO_THROW("Unsupported XeTLA memory layout!");
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
            OPENVINO_THROW("Unsupported XeTLA data type!");
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
                bool is_M_dynamic = eltwise_layout.get_partial_shape()[0].is_dynamic() || eltwise_layout.get_partial_shape()[1].is_dynamic();
                if (!is_M_dynamic) {
                    const auto eltwise_M = extract_channel(ChannelName::BATCH, eltwise_layout) * extract_channel(ChannelName::FEATURE, eltwise_layout);
                    broadcast = eltwise_M == 1;
                }
                assert(eltwise->broadcast_spec.m_axis == 0);

                if (broadcast) {
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

public:
    struct LoraShapeUtils {
        static std::tuple<size_t, size_t, size_t, size_t> get_lora_gemm_shape(const RuntimeParams& params) {
            return {get_total_tokens(params), get_lora_rank(params), get_hidden_size_input(params), get_hidden_size_output(params)};
        }
        static std::tuple<size_t, size_t, size_t, size_t> get_lora_gemm_shape(const cldnn::primitive_inst& instance) {
            return {get_total_tokens(instance), get_lora_rank(instance), get_hidden_size_input(instance), get_hidden_size_output(instance)};
        }

    private:
        static size_t get_total_tokens(const cldnn::layout& layout) {
            assert(layout.format == cldnn::format::bfyx);
            assert(!(layout.get_partial_shape()[0].is_dynamic() || layout.get_partial_shape()[1].is_dynamic()));
            return extract_channel(ChannelName::BATCH, layout) * extract_channel(ChannelName::FEATURE, layout);
        }

        static size_t get_hidden_size_input(const cldnn::layout& layout) {
            assert(layout.format == cldnn::format::bfyx);
            assert(!(layout.get_partial_shape()[2].is_dynamic() || layout.get_partial_shape()[3].is_dynamic()));
            return extract_channel(ChannelName::Y, layout) * extract_channel(ChannelName::X, layout);
        }

        static size_t get_hidden_size_output(const cldnn::layout& layout) {
            assert(layout.format == cldnn::format::bfyx);
            assert(!(layout.get_partial_shape()[2].is_dynamic() || layout.get_partial_shape()[3].is_dynamic()));
            return extract_channel(ChannelName::Y, layout) * extract_channel(ChannelName::X, layout);
        }

        static size_t get_lora_rank(const cldnn::layout& layout) {
            assert(layout.format == cldnn::format::bfyx);
            assert(!(layout.get_partial_shape()[0].is_dynamic()));
            return extract_channel(ChannelName::FEATURE, layout);
        }

    public:
        static size_t get_total_tokens(const RuntimeParams& params) {
            return get_total_tokens(params.output_layouts[0]);
        }

        static size_t get_hidden_size_input(const RuntimeParams& params) {
            return get_hidden_size_input(params.input_layouts[1]);
        }

        static size_t get_hidden_size_output(const RuntimeParams& params) {
            return get_hidden_size_output(params.output_layouts[0]);
        }

        static size_t get_lora_rank(const RuntimeParams& params) {
            return get_lora_rank(params.input_layouts[3]);
        }

        static size_t get_total_tokens(const cldnn::primitive_inst& instance) {
            return get_total_tokens(instance.get_output_layout(0));
        }

        static size_t get_hidden_size_input(const cldnn::primitive_inst& instance) {
            return get_hidden_size_input(instance.get_input_layout(1));
        }

        static size_t get_hidden_size_output(const cldnn::primitive_inst& instance) {
            return get_hidden_size_output(instance.get_output_layout(0));
        }

        static size_t get_lora_rank(const cldnn::primitive_inst& instance) {
            return get_lora_rank(instance.get_input_layout(3));
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
    const Tiling tilingA;
    const Tiling tilingB;
    const size_t total_wg_n_b = 512;

public:
    XetlaLoRAFusedGenerator(Tiling tilingA, Tiling tilingB, std::string_view prefix = "")
        : XeTLALoraBaseGenerator("xetla_lora_fused", prefix),
          tilingA{tilingA},
          tilingB{tilingB} {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto mem_layout_a = Layouts::mem_layout_a;
        const auto mem_layout_state_a = Layouts::mem_layout_state_a;
        const auto mem_layout_state_b = Layouts::mem_layout_state_b;
        const auto mem_layout_c = Layouts::mem_layout_c;

        const uint32_t temp_in_reg = 1;

        jit.add({make_jit_constant("KERNEL_NAME", get_entry_point(params)),
                 make_jit_constant("LORA_DTYPE_A", ov_to_xetla_dtype(params.input_layouts[1].data_type)),
                 make_jit_constant("LORA_DTYPE_B", ov_to_xetla_dtype(params.input_layouts[2].data_type)),
                 make_jit_constant("LORA_DTYPE_C", ov_to_xetla_dtype(params.output_layouts[0].data_type)),
                 make_jit_constant("LORA_DTYPE_ACC", ov_to_xetla_dtype(ov::element::Type_t::f32)),
                 make_jit_constant("LORA_SIZE_RANK", LoraShapeUtils::get_lora_rank_jit(params)),
                 make_jit_constant("LORA_WG_M", tilingA.wg_m),
                 make_jit_constant("LORA_WG_N_A", tilingA.wg_n),
                 make_jit_constant("LORA_WG_N_B", tilingB.wg_n),
                 make_jit_constant("LORA_SG_M", tilingA.sg_m),
                 make_jit_constant("LORA_SG_N_A", tilingA.sg_n),
                 make_jit_constant("LORA_SG_N_B", tilingB.sg_n),
                 make_jit_constant("LORA_SG_K_A", tilingA.sg_k),
                 make_jit_constant("LORA_SG_K_B", tilingB.sg_k),
                 make_jit_constant("LORA_WG_B_TOTAL", total_wg_n_b),
                 make_jit_constant("LORA_LOCAL_SLICING", tilingA.num_local_kslicing),
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
        return DispatchDataFunc{
            [tilingA = tilingA, tilingB = tilingB, total_wg_n_b = total_wg_n_b](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
                assert(!params.is_dynamic());
                auto& wgs = kd.params.workGroups;

                size_t local_range_m = (tilingA.wg_m + tilingA.sg_m - 1) / tilingA.sg_m;
                size_t local_range_nA = (tilingA.wg_n + tilingA.sg_n - 1) / tilingA.sg_n;
                size_t local_range_nB = (tilingB.wg_n + tilingB.sg_n - 1) / tilingB.wg_n;
                size_t local_range_n = local_range_nA > local_range_nB ? local_range_nA : local_range_nB;

                size_t group_range_m = (LoraShapeUtils::get_total_tokens(params) + tilingA.wg_m - 1) / tilingA.wg_m;
                size_t group_range_n = (LoraShapeUtils::get_hidden_size_output(params) + total_wg_n_b - 1) / total_wg_n_b;

                wgs.global = {group_range_n * local_range_n, group_range_m * local_range_m, tilingA.num_global_kslicing * tilingB.num_global_kslicing};
                wgs.local = {local_range_n, local_range_m, tilingA.num_local_kslicing};
            }};
    }
};

class XetlaLoRAGEMMAGenerator : public XeTLALoraBaseGenerator {
    const bool is_aligned;
    const Tiling tiling;

public:
    XetlaLoRAGEMMAGenerator(bool is_aligned, Tiling tiling, std::string_view prefix = "A")
        : XeTLALoraBaseGenerator("xetla_lora_gemmA", prefix),
          is_aligned{is_aligned},
          tiling{tiling} {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto mem_layout_a = Layouts::mem_layout_a;
        const auto mem_layout_b = Layouts::mem_layout_state_a;
        const auto mem_layout_c = Layouts::mem_layout_c;

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
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // temp
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // acc
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // cnt
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});            // state_alpha

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[tiling = tiling](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            size_t group_range_m = (LoraShapeUtils::get_total_tokens(params) + tiling.wg_m - 1u) / tiling.wg_m;
            size_t group_range_n = (LoraShapeUtils::get_lora_rank(params) + tiling.wg_n - 1u) / tiling.wg_n;

            size_t local_range_m = (tiling.wg_m + tiling.sg_m - 1u) / tiling.sg_m;
            size_t local_range_n = (tiling.wg_n + tiling.sg_n - 1u) / tiling.sg_n;

            wgs.global = {group_range_n * local_range_n, group_range_m * local_range_m, tiling.num_global_kslicing * tiling.num_local_kslicing};
            wgs.local = {local_range_n, local_range_m, tiling.num_local_kslicing};
        }};
    }
};

class XetlaLoRAGEMMBGenerator : public XeTLALoraBaseGenerator {
    const bool is_aligned;
    const Tiling tiling;

public:
    XetlaLoRAGEMMBGenerator(bool is_aligned, Tiling tiling, std::string_view prefix = "B")
        : XeTLALoraBaseGenerator("xetla_lora_gemmB", prefix),
          is_aligned{is_aligned},
          tiling{tiling} {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto mem_layout_a = Layouts::mem_layout_a;
        const auto mem_layout_b = Layouts::mem_layout_state_b;
        const auto mem_layout_c = Layouts::mem_layout_c;

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
        return DispatchDataFunc{[tiling = tiling](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            size_t group_range_m = (LoraShapeUtils::get_total_tokens(params) + tiling.wg_m - 1u) / tiling.wg_m;
            size_t group_range_n = (LoraShapeUtils::get_hidden_size_output(params) + tiling.wg_n - 1u) / tiling.wg_n;

            size_t local_range_m = (tiling.wg_m + tiling.sg_m - 1u) / tiling.sg_m;
            size_t local_range_n = (tiling.wg_n + tiling.sg_n - 1u) / tiling.sg_n;

            wgs.global = {group_range_n * local_range_n, group_range_m * local_range_m, tiling.num_global_kslicing * tiling.num_local_kslicing};
            wgs.local = {local_range_n, local_range_m, tiling.num_local_kslicing};
        }};
    }
};

class LoRAImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::LoRAImpl)

    using lora = XeTLALoraBaseGenerator;

    Stage::Ptr lora_fused_short_r32 =  // wg_m, wg_n, sg_m, sg_n, sg_k, global, local
        make_stage<XetlaLoRAFusedGenerator>(lora::Tiling{8, 32, 8, 32, 32, 1, 1}, lora::Tiling{8, 128, 8, 32, 32, 1, 1}, "short_r32");
    Stage::Ptr lora_fused_short_r64 =
        make_stage<XetlaLoRAFusedGenerator>(lora::Tiling{8, 64, 8, 64, 32, 1, 1}, lora::Tiling{8, 128, 8, 32, 32, 1, 1}, "short_r64");
    Stage::Ptr lora_fused_short_r128 =
        make_stage<XetlaLoRAFusedGenerator>(lora::Tiling{8, 128, 8, 128, 32, 1, 1}, lora::Tiling{8, 128, 8, 32, 32, 1, 1}, "short_r128");

    Stage::Ptr lora_fused_long_r32 =
        make_stage<XetlaLoRAFusedGenerator>(lora::Tiling{32 * 4, 32, 32, 32, 32, 1, 1}, lora::Tiling{32 * 4, 128, 32, 32, 32, 1, 1}, "long_r32");
    Stage::Ptr lora_fused_long_r64 =
        make_stage<XetlaLoRAFusedGenerator>(lora::Tiling{16 * 4, 64, 16, 64, 32, 1, 1}, lora::Tiling{16 * 4, 128, 16, 32, 32, 1, 1}, "long_r64");
    Stage::Ptr lora_fused_long_r128 =
        make_stage<XetlaLoRAFusedGenerator>(lora::Tiling{8 * 4, 128, 8, 128, 32, 1, 1}, lora::Tiling{8 * 4, 128, 8, 32, 32, 1, 1}, "long_128");

    Stage::Ptr lora_gemm_a_short_slicing1 = make_stage<XetlaLoRAGEMMAGenerator>(true, lora::Tiling{8, 32, 8, 16, 32, 1, 1}, "a_short_s1");
    Stage::Ptr lora_gemm_a_short_slicing8 = make_stage<XetlaLoRAGEMMAGenerator>(true, lora::Tiling{8, 32, 8, 16, 32, 1, 8}, "a_short_s8");

    Stage::Ptr lora_gemm_a_long0_slicing1 = make_stage<XetlaLoRAGEMMAGenerator>(true, lora::Tiling{128, 32, 32, 16, 32, 1, 1}, "a_long0_s1");
    Stage::Ptr lora_gemm_a_long0_slicing2 = make_stage<XetlaLoRAGEMMAGenerator>(true, lora::Tiling{128, 32, 32, 16, 32, 1, 2}, "a_long0_s2");

    Stage::Ptr lora_gemm_b_short = make_stage<XetlaLoRAGEMMBGenerator>(true, lora::Tiling{8, 128, 8, 16, 32, 1, 1}, "b_short");
    Stage::Ptr lora_gemm_b_long0 = make_stage<XetlaLoRAGEMMBGenerator>(true, lora::Tiling{128, 256, 32, 32, 32, 1, 1}, "b_long0");

    Stage::Ptr lora_gemm_a_unaligned = make_stage<XetlaLoRAGEMMAGenerator>(false, lora::Tiling{8 * 8, 16 * 4, 8, 16, 32, 1, 1}, "a_unaligned");
    Stage::Ptr lora_gemm_b_unaligned = make_stage<XetlaLoRAGEMMBGenerator>(false, lora::Tiling{8 * 4, 16 * 8, 8, 16, 32, 1, 1}, "b_unaligned");

    LoRAImpl() : PrimitiveImplOCL(LoRAImplementationManager::get_type_info_static()) {}
    LoRAImpl(const program_node& node, const RuntimeParams& params) : LoRAImpl() {
        add_stage(lora_fused_short_r32, params);
        add_stage(lora_fused_short_r64, params);
        add_stage(lora_fused_short_r128, params);
        add_stage(lora_fused_long_r32, params);
        add_stage(lora_fused_long_r64, params);
        add_stage(lora_fused_long_r128, params);
        add_stage(lora_gemm_a_short_slicing1, params);
        add_stage(lora_gemm_a_short_slicing8, params);
        add_stage(lora_gemm_a_long0_slicing1, params);
        add_stage(lora_gemm_a_long0_slicing2, params);
        add_stage(lora_gemm_b_short, params);
        add_stage(lora_gemm_b_long0, params);
        add_stage(lora_gemm_a_unaligned, params);
        add_stage(lora_gemm_b_unaligned, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<LoRAImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        size_t buf_size = XeTLALoraBaseGenerator::LoraShapeUtils::get_total_tokens(params) * XeTLALoraBaseGenerator::LoraShapeUtils::get_lora_rank(params);
        return {BufferDescriptor{buf_size, ov::element::f16}, BufferDescriptor{0, ov::element::f32}, BufferDescriptor{0, ov::element::u32}};
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        cldnn::stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }

        update_rt_params(instance);

        std::vector<cldnn::event::ptr> tmp_events(events);
        const auto exec_stages = get_stages_execution_order(instance);
        for (const auto& stage_id : exec_stages) {
            tmp_events = {execute_stage(tmp_events, instance, *_stages[stage_id])};
        }

        return tmp_events[0];
    }

private:
    enum KernelsTypes {
        FUSED_SHORT_R32 = 0,
        FUSED_SHORT_R64,
        FUSED_SHORT_R128,
        FUSED_LONG_R32,
        FUSED_LONG_R64,
        FUSED_LONG_R128,
        GEMM_A_SHORT_S1,
        GEMM_A_SHORT_S8,
        GEMM_A_LONG0_S1,
        GEMM_A_LONG0_S2,
        GEMM_B_SHORT,
        GEMM_B_LONG0,
        GEMM_A_UNALIGNED,
        GEMM_B_UNALIGNED
    };

    std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) const override {
        std::vector<size_t> stages_order;
        using lora = XeTLALoraBaseGenerator;

        bool is_empty_lora = instance.get_input_layout(2).count() == 0;
        if (is_empty_lora) {
            assert(!instance.has_fused_primitives());
            return stages_order;
        }

        auto [tokens, rank, hidden_in, hidden_out] = lora::LoraShapeUtils::get_lora_gemm_shape(instance);

        const auto ld_input = lora::Layouts::mem_layout_a == lora::MemLayout::col_major ? tokens : hidden_in;
        const auto ld_state_a = lora::Layouts::mem_layout_state_a == lora::MemLayout::col_major ? hidden_out : rank;
        const auto ld_state_b = lora::Layouts::mem_layout_state_b == lora::MemLayout::col_major ? rank : hidden_in;
        const auto ld_state_temp = lora::Layouts::mem_layout_temp == lora::MemLayout::col_major ? tokens : rank;
        const auto ld_state_output = lora::Layouts::mem_layout_c == lora::MemLayout::col_major ? tokens : hidden_out;

        const bool is_aligned_input = lora::is_2dload_aligned(ld_input, instance.get_input_layout(1).data_type);
        const bool is_aligned_state_a = lora::is_2dload_aligned(ld_state_a, instance.get_input_layout(2).data_type);
        const bool is_aligned_state_b = lora::is_2dload_aligned(ld_state_b, instance.get_input_layout(4).data_type);
        const bool is_aligned_temp = lora::is_2dload_aligned(ld_state_temp, instance.get_input_layout(1).data_type);
        const bool is_aligned_output = lora::is_2dload_aligned(ld_state_output, instance.get_output_layout(0).data_type);

        const bool can_use_fused_reg = rank <= 128 && is_aligned_input && is_aligned_state_a && is_aligned_state_b && is_aligned_output;
        const bool is_gemmA_aligned = is_aligned_input && is_aligned_state_a && is_aligned_temp;
        const bool is_gemmB_aligned = is_aligned_temp && is_aligned_state_b && is_aligned_output;

        if (tokens <= 32 && is_gemmA_aligned && is_gemmB_aligned) {
            size_t iters = (hidden_in + 32 - 1) / 32;
            if (iters > 16) {
                stages_order.emplace_back(KernelsTypes::GEMM_A_SHORT_S8);
            } else {
                stages_order.emplace_back(KernelsTypes::GEMM_A_SHORT_S1);
            }
            stages_order.emplace_back(KernelsTypes::GEMM_B_SHORT);
            return stages_order;
        }

        if (tokens > 32 && is_gemmA_aligned && is_gemmB_aligned) {
            size_t iters = (hidden_in + 32 - 1) / 32;
            if (iters > 4) {
                stages_order.emplace_back(KernelsTypes::GEMM_A_LONG0_S2);
            } else {
                stages_order.emplace_back(KernelsTypes::GEMM_A_LONG0_S1);
            }
            stages_order.emplace_back(KernelsTypes::GEMM_B_LONG0);
            return stages_order;
        }

        if (can_use_fused_reg) {
            if (tokens <= 32) {
                KernelsTypes kernel_type = KernelsTypes::FUSED_SHORT_R32;
                if (rank <= 128) {
                    kernel_type = KernelsTypes::FUSED_SHORT_R128;
                }
                if (rank <= 64) {
                    kernel_type = KernelsTypes::FUSED_SHORT_R64;
                }
                if (rank <= 32) {
                    kernel_type = KernelsTypes::FUSED_SHORT_R32;
                }
                stages_order.emplace_back(kernel_type);
            } else {
                KernelsTypes kernel_type = KernelsTypes::FUSED_LONG_R32;
                if (rank <= 128) {
                    kernel_type = KernelsTypes::FUSED_LONG_R128;
                }
                if (rank <= 64) {
                    kernel_type = KernelsTypes::FUSED_LONG_R64;
                }
                if (rank <= 32) {
                    kernel_type = KernelsTypes::FUSED_LONG_R32;
                }
                stages_order.emplace_back(kernel_type);
            }
            return stages_order;
        }

        stages_order.emplace_back(KernelsTypes::GEMM_A_UNALIGNED);
        stages_order.emplace_back(KernelsTypes::GEMM_B_UNALIGNED);
        return stages_order;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> LoRAImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<lora>());
    return std::make_unique<LoRAImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::LoRAImpl)
