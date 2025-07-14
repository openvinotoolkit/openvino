// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora.hpp"

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

enum LoraType { SINGLE = 0, HORIZONTAL_FUSED };

enum SingleKernelTypes {
    REFERENCE = 0,
    FIRST_TOKEN_A_REF,
    FIRST_TOKEN_A_SMALL,
    FIRST_TOKEN_A_MEDIUM,
    FIRST_TOKEN_A_LARGE,
    FIRST_TOKEN_B_REF,
    FIRST_TOKEN_B_MEDIUM,
    FIRST_TOKEN_B_LARGE,
    SECOND_TOKEN_A,
    SECOND_TOKEN_B,
    FUSED_OPS
};

enum FusedKernelTypes {
    HF_REFERENCE = 11,
    HF_FIRST_TOKEN_A_REF,
    HF_FIRST_TOKEN_A_SMALL,
    HF_FIRST_TOKEN_A_MEDIUM,
    HF_FIRST_TOKEN_A_LARGE,
    HF_FIRST_TOKEN_B_REF,
    HF_FIRST_TOKEN_B_SMALL,
    HF_FIRST_TOKEN_B_MEDIUM,
    HF_FIRST_TOKEN_B_LARGE,
    HF_SECOND_TOKEN_A,
    HF_SECOND_TOKEN_B
};

template <LoraType LT = LoraType::SINGLE>
class LoraRefBase : public KernelGenerator {
protected:
    explicit LoraRefBase(std::string_view name, std::string_view suffix = "") : KernelGenerator(name, suffix) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<lora>();

        jit_constants.add(make_type_jit_constants("ACCUMULATOR", params.get_input_layout(0).data_type));
        jit_constants.add(make_type_jit_constants("STATE", params.get_input_layout(2).data_type));

        jit_constants.make("MAX_LORA_RANK", 256);

        LayoutJitter alpha_state_dims(params.input_layouts[3], params.in_port_to_shape_info_offset.at(3));
        jit_constants.make("LORA_RANK", alpha_state_dims.dim(ChannelName::FEATURE));

        jit_constants.make("LORA_COUNT", get_lora_count(params));

        return jit_constants;
    }

public:
    [[nodiscard]] static size_t get_lora_count(const RuntimeParams& params) {
        auto desc = params.typed_desc<lora>();
        return (desc->input_size() - 2ul) / 3ul;
    }
};

template <LoraType LT>
class LoraRef : public LoraRefBase<LT> {
public:
    LoraRef() : LoraRefBase<LT>("lora_ref", LT == LoraType::SINGLE ? "" : "hf") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraRefBase<LT>::get_jit_constants(params);

        if (LT == LoraType::SINGLE) {
            jit_constants.make("BASE_KERNEL", 1);
        } else {
            jit_constants.make("HORIZONTAL_FUSED", 1);
        }
        return jit_constants;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& state_alpha_lo = params.input_layouts[3];
            size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);
            if (lora_rank == 0) {
                return;
            }

            const auto& out_l = params.output_layouts[0];
            size_t b = extract_channel(ChannelName::BATCH, out_l);
            size_t f = extract_channel(ChannelName::FEATURE, out_l);
            size_t y = extract_channel(ChannelName::Y, out_l);
            size_t x = extract_channel(ChannelName::X, out_l);

            wgs.global = {b * f, align_to(y * x, lora_rank), 1};
            wgs.local = {1, lora_rank, 1};
        }};
    }
};

class LoraFusedOps : public KernelGenerator {
public:
    LoraFusedOps() : KernelGenerator("lora_ref", "fused_ops") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);

        jit_constants.make("FUSED_OPS_KERNEL", 1);

        if (params.has_fused_primitives()) {
            const auto& out_l = params.get_output_layout(0);
            FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "output[output_idx]", out_l.data_type};
            jit_constants.add(make_fused_ops_jit_constants(params, {conf}));
        }

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        add_fused_ops_arguments(args, params);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            std::vector<std::vector<ChannelName>> dims_by_gws = {{ChannelName::BATCH}, {ChannelName::FEATURE}, {ChannelName::Y, ChannelName::X}};

            const auto& out_l = params.output_layouts[0];
            size_t b = extract_channel(ChannelName::BATCH, out_l);
            size_t f = extract_channel(ChannelName::FEATURE, out_l);
            size_t y = extract_channel(ChannelName::Y, out_l);
            size_t x = extract_channel(ChannelName::X, out_l);

            wgs.global = {b, f, y * x};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), out_l.format, out_l.format, dims_by_gws);
        }};
    }
};

template <LoraType LT = LoraType::SINGLE>
class LoraOptBase : public LoraRefBase<LT> {
protected:
    explicit LoraOptBase(std::string_view suffix = "") : LoraRefBase<LT>(LT == LoraType::SINGLE ? "lora_opt" : "lora_horizontal_fused", suffix) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraRefBase<LT>::get_jit_constants(params);
        auto desc = params.typed_desc<lora>();

        size_t max_workgroup_size = params.get_device_info().max_work_group_size;
        jit_constants.make("MAX_WORKGROUP_SIZE", max_workgroup_size);

        jit_constants.make("SUBGROUP_SIZE", get_subgroup_size(params));

        return jit_constants;
    }

public:
    [[nodiscard]] static size_t get_subgroup_size(const RuntimeParams& params) {
        auto in_dtype = params.get_input_layout().data_type;
        size_t subgroup_size = in_dtype == ov::element::f16 ? 16ul : 8ul;
        return subgroup_size;
    }

    [[nodiscard]] static size_t get_max_gemma_sgk(const RuntimeParams& params) {
        size_t max_workgroup_size = params.get_device_info().max_work_group_size;
        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);
        size_t max_gemma_sgk = max_workgroup_size / min_lora_rank / lora_count;
        return max_gemma_sgk;
    }

    [[nodiscard]] static size_t is_first_token(const RuntimeParams& params) {
        const auto& lora_input_lo = params.input_layouts[1];
        size_t batch = extract_channel(ChannelName::BATCH, lora_input_lo);
        size_t feature = extract_channel(ChannelName::FEATURE, lora_input_lo);
        return batch * feature > 1;
    }

    static constexpr size_t gemm_a_sg_bk = 32ul;
    static constexpr size_t min_lora_rank = 16ul;
};

template <LoraType LT>
class LoraOptFirstTokenBase : public LoraOptBase<LT> {
protected:
    explicit LoraOptFirstTokenBase(std::string_view suffix, size_t reg_m, size_t reg_n) : LoraOptBase<LT>(suffix), reg_m(reg_m), reg_n(reg_n) {}

    // clang-format off

    std::string generate_block_read(ov::element::Type_t dtype, std::string input) const {
        std::string res = dtype == ov::element::f16 ? "intel_sub_group_block_read_us((const __global ushort*)("
                                                    : "intel_sub_group_block_read((const __global uint*)(";
        res += input + "))";
        return res;
    }

    std::string generate_block_write(ov::element::Type_t dtype, std::string dst, std::string src) const {
        std::string res = "";
        if (dtype == ov::element::f16) {
            res = "intel_sub_group_block_write_us((__global ushort*)(" + dst + "), as_short(" + src + "));";
        } else {
            res = "intel_sub_group_block_write((__global uint*)(" + dst + "), as_int(" + src + "));";
        }
        return res;
    }

    std::string generate_broadcast(ov::element::Type_t dtype, std::string input) const {
        std::string res = dtype == ov::element::f16 ? "intel_sub_group_broadcast("
                                                    : "sub_group_broadcast(";
        res += input + ")";
        return res;
    }

    std::string generate_matmul_code(size_t M, size_t N, ov::element::Type_t dtype, size_t lora_count, bool is_a_kernel) const {
        std::string res = "";
        std::string int_type = dtype == ov::element::f16 ? "ushort" : "uint";
        std::string input_type = is_a_kernel ? "INPUT1_TYPE" : "ACCUMULATOR_TYPE";

        std::string K_dim = lora_count == 1 ? "K" : "strideA";
        std::string N_dim = lora_count == 1 ? "N" : "strideB";

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                res += "ACCUMULATOR_TYPE sum_" + std::to_string(m) + "_" + std::to_string(n) + " = 0;";
            }
        }

        res += "for (int i = 0; i < K; i += SUBGROUP_SIZE) {";

        for (size_t m = 0; m < M; ++m) {
            res += int_type + " input_" + std::to_string(m) + " = " + generate_block_read(dtype, "ptrA + " + std::to_string(m) + " * " + K_dim) + ";";
        }

        res += "for (int kk = 0; kk < SUBGROUP_SIZE; kk++) {";

        for (size_t n = 0; n < N; ++n) {
            res += "ACCUMULATOR_TYPE bb_" + std::to_string(n) + " = "
                + "TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(" + generate_block_read(dtype, "ptrB + " + std::to_string(n) + " * SUBGROUP_SIZE") + "));";
        }

        for (size_t m = 0; m < M; ++m) {
            res += "ACCUMULATOR_TYPE aa_" + std::to_string(m) + " = "
                + "TO_ACCUMULATOR_TYPE(AS_" + input_type + "(" + generate_broadcast(dtype, "input_" + std::to_string(m) + ", kk") + "));";
        }

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                std::string sum_var = "sum_" + std::to_string(m) + "_" + std::to_string(n);
                res += sum_var + " = fma(aa_" + std::to_string(m) + ", bb_" + std::to_string(n) + ", " + sum_var + ");";
            }
        }

        res += "ptrB += " + N_dim + "; }";
        res += "ptrA += SUBGROUP_SIZE; }";

        return res;
    }

    std::string generate_store_result_code(size_t M, size_t N, ov::element::Type_t dtype, bool is_a_kernel) const {
        std::string res = "";

        if (is_a_kernel) {
            for (size_t n = 0; n < N; ++n) {
                res += "ACCUMULATOR_TYPE alpha_" + std::to_string(n) + " = "
                    + "TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(" + generate_block_read(dtype, "alpha_ptr + " + std::to_string(n) + " * SUBGROUP_SIZE") + "));";
            }

            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    std::string scaled_alpha = "(alpha_" + std::to_string(n) + " / TO_ACCUMULATOR_TYPE(LORA_RANK))";
                    res += generate_block_write(dtype,
                                                "ptrC + SUBGROUP_SIZE * " + std::to_string(n),
                                                "sum_" + std::to_string(m) + "_" + std::to_string(n) + " * " + scaled_alpha);
                }
                res += "ptrC += N;";
            }
        } else {
            for (size_t n = 0; n < N; ++n) {
                res += "INPUT0_TYPE main_N_" + std::to_string(n) + " = 0;";
            }

            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    res += "main_N_" + std::to_string(n) + " = "
                        + "AS_INPUT0_TYPE(" + generate_block_read(dtype, "main_ptr + " + std::to_string(n) + " * SUBGROUP_SIZE") + ");";

                    res += generate_block_write(dtype,
                                                "ptrC + " + std::to_string(n) + " * SUBGROUP_SIZE",
                                                "TO_INPUT0_TYPE(sum_" + std::to_string(m) + "_" + std::to_string(n) + ") + main_N_" + std::to_string(n));
                }
                res += "main_ptr += N;";
                res += "ptrC += N;";
            }
        }

        return res;
    }

    // clang-format on

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptBase<LT>::get_jit_constants(params);

        jit_constants.make("REG_M", reg_m);
        jit_constants.make("REG_N", reg_n);

        LayoutJitter lora_input(params.input_layouts[1], params.in_port_to_shape_info_offset.at(1));
        auto b = lora_input.dim(ChannelName::BATCH);
        auto f = lora_input.dim(ChannelName::FEATURE);
        jit_constants.make("M", "(" + b + " * " + f + ")");

        return jit_constants;
    }

    size_t reg_m;
    size_t reg_n;
};

template <LoraType LT>
class LoraOptFirstTokenA : public LoraOptFirstTokenBase<LT> {
public:
    explicit LoraOptFirstTokenA(std::string suffix, size_t reg_m, size_t reg_n) : LoraOptFirstTokenBase<LT>("first_token_a_" + suffix, reg_m, reg_n) {}

protected:
    using LoraOptFirstTokenBase<LT>::reg_m;
    using LoraOptFirstTokenBase<LT>::reg_n;

    static std::pair<size_t, size_t> get_subgroup_params(const RuntimeParams& params, size_t reg_n) {
        size_t sg_m = 0, sg_n = 0;

        const auto& state_alpha_lo = params.input_layouts[3];
        size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);

        size_t subgroup_size = LoraOptBase<LT>::get_subgroup_size(params);
        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);
        size_t max_sg_num = params.get_device_info().max_work_group_size / subgroup_size;

        sg_n = lora_rank * lora_count / subgroup_size / reg_n;
        if (sg_n != 0) {
            if (lora_rank == 1) {
                sg_m = ceil_div(subgroup_size, sg_n);
            } else {
                sg_m = max_sg_num / sg_n;
            }
        }

        return {sg_m, sg_n};
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptFirstTokenBase<LT>::get_jit_constants(params);

        jit_constants.make("FIRST_TOKEN_A", 1);

        jit_constants.make("N", "LORA_RANK * LORA_COUNT");

        LayoutJitter state_a_dims(params.input_layouts[2], params.in_port_to_shape_info_offset.at(2));
        jit_constants.make("K", state_a_dims.dim(ChannelName::FEATURE));

        auto in_dtype = params.get_input_layout().data_type;
        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);
        jit_constants.make("MAIN_MATMUL_CODE", LoraOptFirstTokenBase<LT>::generate_matmul_code(reg_m, reg_n, in_dtype, lora_count, true));

        jit_constants.make("MULTIPLY_AND_STORE_CODE", LoraOptFirstTokenBase<LT>::generate_store_result_code(reg_m, reg_n, in_dtype, true));

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});

        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

        if (lora_count > 1) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 5});
            args.push_back({ArgumentDescriptor::Types::INPUT, 6});
        }

        if (lora_count > 2) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 8});
            args.push_back({ArgumentDescriptor::Types::INPUT, 9});
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[reg_m = reg_m, reg_n = reg_n](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& lora_input_lo = params.input_layouts[1];
            size_t batch = extract_channel(ChannelName::BATCH, lora_input_lo);
            size_t feature = extract_channel(ChannelName::FEATURE, lora_input_lo);

            const auto& state_alpha_lo = params.input_layouts[3];
            size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);

            size_t subgroup_size = LoraOptBase<LT>::get_subgroup_size(params);
            size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

            size_t sg_m = 0, sg_n = 0;
            std::tie(sg_m, sg_n) = LoraOptFirstTokenA::get_subgroup_params(params, reg_n);

            if (sg_n == 0) {
                return;
            }

            size_t bm = reg_m * sg_m;
            size_t bn = reg_n * sg_n * subgroup_size;

            wgs.global = {round_up_to(batch * feature, bm) / reg_m, round_up_to(lora_rank * lora_count, bn) / reg_n, 1};
            wgs.local = {sg_m, sg_n * subgroup_size, 1};
        }};
    }
};

template <LoraType LT>
class LoraOptFirstTokenB : public LoraOptFirstTokenBase<LT> {
public:
    explicit LoraOptFirstTokenB(std::string suffix, size_t reg_m, size_t reg_n, size_t sg_m)
        : LoraOptFirstTokenBase<LT>("first_token_b_" + suffix, reg_m, reg_n),
          sg_m(sg_m) {}

protected:
    using LoraOptFirstTokenBase<LT>::reg_m;
    using LoraOptFirstTokenBase<LT>::reg_n;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptFirstTokenBase<LT>::get_jit_constants(params);

        jit_constants.make("FIRST_TOKEN_B", 1);

        LayoutJitter state_b_dims(params.input_layouts[4], params.in_port_to_shape_info_offset.at(4));
        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

        if (lora_count == 1) {
            jit_constants.make("N", state_b_dims.dim(ChannelName::BATCH));
        } else {
            jit_constants.make("N0", state_b_dims.dim(ChannelName::BATCH));

            LayoutJitter state_b_kv_dims(params.input_layouts[7], params.in_port_to_shape_info_offset.at(7));
            if (lora_count == 2) {
                jit_constants.make("N", "2 * " + state_b_kv_dims.dim(ChannelName::BATCH));
            } else {
                jit_constants.make("N1_2", state_b_kv_dims.dim(ChannelName::BATCH));
                jit_constants.make("N", "(N0 + 2 * N1_2)");
            }
        }

        jit_constants.make("K", "LORA_RANK");

        auto in_dtype = params.get_input_layout().data_type;
        jit_constants.make("MAIN_MATMUL_CODE", LoraOptFirstTokenBase<LT>::generate_matmul_code(reg_m, reg_n, in_dtype, lora_count, false));

        jit_constants.make("ADD_AND_STORE_CODE", LoraOptFirstTokenBase<LT>::generate_store_result_code(reg_m, reg_n, in_dtype, false));

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});

        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

        if (lora_count > 1) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 7});
        }

        if (lora_count > 2) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 10});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[reg_m = reg_m, reg_n = reg_n, sg_m = sg_m](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& lora_input_lo = params.input_layouts[1];
            size_t batch = extract_channel(ChannelName::BATCH, lora_input_lo);
            size_t feature = extract_channel(ChannelName::FEATURE, lora_input_lo);

            const auto& state_b_lo = params.input_layouts[4];
            size_t output_state = extract_channel(ChannelName::BATCH, state_b_lo);

            size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

            if (lora_count > 1) {
                size_t kv_state = extract_channel(ChannelName::BATCH, params.input_layouts[7]);
                size_t fused_mlp_output = kv_state * 2;

                if (lora_count == 2) {
                    output_state = fused_mlp_output;
                } else {
                    size_t q_state = extract_channel(ChannelName::FEATURE, params.input_layouts[2]);
                    output_state = fused_mlp_output + q_state;
                }
            }

            size_t subgroup_size = LoraOptBase<LT>::get_subgroup_size(params);
            size_t max_sg_num = params.get_device_info().max_work_group_size / subgroup_size;
            size_t sg_n = max_sg_num / sg_m;

            size_t bm = reg_m * sg_m;
            size_t bn = reg_n * sg_n * subgroup_size;

            wgs.global = {round_up_to(batch * feature, bm) / reg_m, round_up_to(output_state, bn) / reg_n, 1};
            wgs.local = {sg_m, sg_n * subgroup_size, 1};
        }};
    }

    size_t sg_m;
};

template <LoraType LT>
class LoraOptSecondTokenA : public LoraOptBase<LT> {
public:
    LoraOptSecondTokenA() : LoraOptBase<LT>("second_token_a") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptBase<LT>::get_jit_constants(params);

        jit_constants.make("SECOND_TOKEN_A", 1);

        LayoutJitter state_a_dims(params.input_layouts[2], params.in_port_to_shape_info_offset.at(2));
        jit_constants.make("K", state_a_dims.dim(ChannelName::FEATURE));

        jit_constants.make("MAX_GEMMA_SGK", LoraOptBase<LT>::get_max_gemma_sgk(params));

        jit_constants.make("GEMMA_SGK", "min(MAX_WORKGROUP_SIZE / LORA_RANK, MAX_GEMMA_SGK)");

        jit_constants.make("GEMMA_SG_BK", LoraOptBase<LT>::gemm_a_sg_bk);

        if (LT == LoraType::HORIZONTAL_FUSED) {
            jit_constants.make("MAX_GEMMA_N", "MAX_LORA_RANK * LORA_COUNT");
        }

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});

        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

        if (lora_count > 1) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 5});
        }

        if (lora_count > 2) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 8});
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& state_alpha_lo = params.input_layouts[3];
            size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);
            if (lora_rank == 0) {
                return;
            }

            size_t max_workgroup_size = params.get_device_info().max_work_group_size;
            size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

            size_t gemma_sgK = max_workgroup_size / lora_rank;
            size_t lws_2 = lora_rank;

            if (lora_rank * lora_count <= max_workgroup_size) {
                gemma_sgK /= lora_count;
                lws_2 *= lora_count;
            }

            size_t input_state = extract_channel(ChannelName::FEATURE, params.input_layouts[2]);
            size_t gemma_wgs = ceil_div(input_state, LoraOptBase<LT>::gemm_a_sg_bk * gemma_sgK);

            wgs.global = {gemma_wgs, gemma_sgK, lora_rank * lora_count};
            wgs.local = {1, gemma_sgK, lws_2};
        }};
    }
};

template <LoraType LT>
class LoraOptSecondTokenB : public LoraOptBase<LT> {
public:
    LoraOptSecondTokenB() : LoraOptBase<LT>("second_token_b") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptBase<LT>::get_jit_constants(params);

        jit_constants.make("SECOND_TOKEN_B", 1);

        LayoutJitter state_b_dims(params.input_layouts[4], params.in_port_to_shape_info_offset.at(4));

        size_t lora_count = LoraRefBase<>::get_lora_count(params);
        if (lora_count == 1) {
            jit_constants.make("N", state_b_dims.dim(ChannelName::BATCH));
        } else {
            jit_constants.make("N0", state_b_dims.dim(ChannelName::BATCH));
            jit_constants.make("MAX_GEMMA_N", "MAX_LORA_RANK * LORA_COUNT");

            LayoutJitter state_b_kv_dims(params.input_layouts[7], params.in_port_to_shape_info_offset.at(7));
            if (lora_count == 2) {
                jit_constants.make("N", "2 * " + state_b_kv_dims.dim(ChannelName::BATCH));
            } else {
                jit_constants.make("N1_2", state_b_kv_dims.dim(ChannelName::BATCH));
                jit_constants.make("N", "(N0 + 2 * N1_2)");
            }
        }

        LayoutJitter state_a_dims(params.input_layouts[2], params.in_port_to_shape_info_offset.at(2));
        jit_constants.make("K", state_a_dims.dim(ChannelName::FEATURE));

        jit_constants.make("GEMMA_SG_BK", LoraOptBase<LT>::gemm_a_sg_bk);

        jit_constants.make("MAX_GEMMA_SGK", LoraOptBase<LT>::get_max_gemma_sgk(params));

        jit_constants.make("GEMMA_SGK", "min(MAX_WORKGROUP_SIZE / (LORA_RANK * LORA_COUNT), MAX_GEMMA_SGK)");

        jit_constants.make("GEMMB_PART_NUM", "CEIL_DIV(K, GEMMA_SG_BK * GEMMA_SGK)");

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});

        size_t lora_count = LoraRefBase<LT>::get_lora_count(params);

        if (lora_count > 1) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 6});
            args.push_back({ArgumentDescriptor::Types::INPUT, 7});
        }

        if (lora_count > 2) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 9});
            args.push_back({ArgumentDescriptor::Types::INPUT, 10});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            size_t max_workgroup_size = params.get_device_info().max_work_group_size;
            size_t subgroup_size = LoraOptBase<>::get_subgroup_size(params);
            size_t lora_count = LoraRefBase<>::get_lora_count(params);

            if (lora_count == 1) {
                size_t output_state = extract_channel(ChannelName::BATCH, params.input_layouts[4]);

                wgs.global = {round_up_to(output_state, max_workgroup_size), 1, 1};
                wgs.local = {max_workgroup_size, 1, 1};
            } else {
                size_t kv_state = extract_channel(ChannelName::BATCH, params.input_layouts[7]);
                size_t fused_output = kv_state * 2;

                if (lora_count == 3) {
                    size_t q_state = extract_channel(ChannelName::FEATURE, params.input_layouts[2]);
                    fused_output += q_state;
                }

                size_t gemmb_sgN = std::min(fused_output, max_workgroup_size) / subgroup_size;
                size_t gemmb_wg_sz = gemmb_sgN * subgroup_size;

                wgs.global = {round_up_to(fused_output, gemmb_wg_sz), 1, 1};
                wgs.local = {gemmb_wg_sz, 1, 1};
            }
        }};
    }
};

bool is_optimized_kernel_supported(const RuntimeParams& params) {
    size_t subgroup_size = LoraOptBase<>::get_subgroup_size(params);

    const auto& state_a_layout = params.get_input_layout(2);
    size_t input_state = state_a_layout.get_shape().back();
    if (input_state % subgroup_size != 0) {
        return false;
    }

    const auto& alpha_layout = params.get_input_layout(3);
    size_t lora_rank = alpha_layout.get_shape().back();
    if (lora_rank % subgroup_size != 0) {
        return false;
    }

    const auto& state_b_layout = params.get_input_layout(4);
    size_t output_state = state_b_layout.get_shape().front();
    if (output_state % subgroup_size != 0) {
        return false;
    }

    return true;
}

std::vector<size_t> get_stages_execution_order_single_lora(const cldnn::primitive_inst& instance) {
    std::vector<size_t> stages_order;
    const auto& params = *instance.get_impl_params();

    bool is_empty_lora = instance.get_input_layout(2).count() == 0;
    if (!is_empty_lora) {
        if (!is_optimized_kernel_supported(params)) {
            stages_order.emplace_back(SingleKernelTypes::REFERENCE);
        } else {
            if (LoraOptBase<>::is_first_token(params)) {
                const auto& state_alpha_lo = instance.get_input_layout(3);
                size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);

                const auto& lora_input = instance.get_input_layout(1);
                size_t batch = extract_channel(ChannelName::BATCH, lora_input);
                size_t feature = extract_channel(ChannelName::FEATURE, lora_input);
                size_t acc_batch = batch * feature;

                if ((lora_rank == 128 || lora_rank == 256) && acc_batch >= 16) {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_A_LARGE);
                } else if (lora_rank == 64 && acc_batch >= 8) {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_A_MEDIUM);
                } else if (acc_batch >= 4) {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_A_SMALL);
                } else {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_A_REF);
                }

                if (acc_batch < 8) {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_B_REF);
                } else if (acc_batch < 256) {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_B_MEDIUM);
                } else {
                    stages_order.emplace_back(SingleKernelTypes::FIRST_TOKEN_B_LARGE);
                }
            } else {
                stages_order.emplace_back(SingleKernelTypes::SECOND_TOKEN_A);
                stages_order.emplace_back(SingleKernelTypes::SECOND_TOKEN_B);
            }
        }
    }

    if (instance.has_fused_primitives()) {
        stages_order.emplace_back(SingleKernelTypes::FUSED_OPS);
    }

    return stages_order;
}

std::vector<size_t> get_stages_execution_order_hf_lora(const cldnn::primitive_inst& instance) {
    const auto& params = *instance.get_impl_params();
    std::vector<size_t> stages_order;

    if (!is_optimized_kernel_supported(params)) {
        return {FusedKernelTypes::HF_REFERENCE};
    }

    const auto& lora_input = instance.get_input_layout(1);
    size_t batch = extract_channel(ChannelName::BATCH, lora_input);
    size_t feature = extract_channel(ChannelName::FEATURE, lora_input);
    size_t acc_batch = batch * feature;
    bool is_first_token = acc_batch > 1;

    if (is_first_token) {
        const auto& state_alpha_lo = instance.get_input_layout(3);
        size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);
        size_t lora_count = LoraRefBase<>::get_lora_count(params);
        size_t max_workgroup_size = params.get_device_info().max_work_group_size;

        if (acc_batch > 1000 && lora_rank > 16) {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_A_LARGE);
        } else if (acc_batch < 8) {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_A_REF);
        } else if (lora_rank * lora_count > max_workgroup_size) {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_A_MEDIUM);
        } else {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_A_SMALL);
        }

        size_t subgroup_size = LoraOptBase<>::get_subgroup_size(params);

        size_t kv_state = extract_channel(ChannelName::BATCH, instance.get_input_layout(7));
        size_t fused_mlp_output = kv_state * 2;

        size_t output_state = 0;
        if (lora_count == 2) {
            output_state = fused_mlp_output;
        } else {
            size_t q_state = extract_channel(ChannelName::FEATURE, instance.get_input_layout(2));
            output_state = fused_mlp_output + q_state;
        }

        if (acc_batch < 8) {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_B_REF);
        } else if (output_state % (2 * subgroup_size) != 0) {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_B_SMALL);
        } else if (acc_batch < 256) {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_B_MEDIUM);
        } else {
            stages_order.emplace_back(FusedKernelTypes::HF_FIRST_TOKEN_B_LARGE);
        }
    } else {
        stages_order.emplace_back(FusedKernelTypes::HF_SECOND_TOKEN_A);
        stages_order.emplace_back(FusedKernelTypes::HF_SECOND_TOKEN_B);
    }

    return stages_order;
}

class LoraImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::LoraImpl)

    Stage::Ptr lora_ref = make_stage<LoraRef<SINGLE>>();
    Stage::Ptr lora_opt_first_token_a_ref = make_stage<LoraOptFirstTokenA<SINGLE>>("ref", 1, 1);
    Stage::Ptr lora_opt_first_token_a_small = make_stage<LoraOptFirstTokenA<SINGLE>>("small", 4, 1);
    Stage::Ptr lora_opt_first_token_a_medium = make_stage<LoraOptFirstTokenA<SINGLE>>("medium", 8, 2);
    Stage::Ptr lora_opt_first_token_a_large = make_stage<LoraOptFirstTokenA<SINGLE>>("large", 16, 2);
    Stage::Ptr lora_opt_first_token_b_ref = make_stage<LoraOptFirstTokenB<SINGLE>>("ref", 1, 2, 8);
    Stage::Ptr lora_opt_first_token_b_medium = make_stage<LoraOptFirstTokenB<SINGLE>>("medium", 8, 2, 16);
    Stage::Ptr lora_opt_first_token_b_large = make_stage<LoraOptFirstTokenB<SINGLE>>("large", 16, 2, 8);
    Stage::Ptr lora_opt_second_token_a = make_stage<LoraOptSecondTokenA<SINGLE>>();
    Stage::Ptr lora_opt_second_token_b = make_stage<LoraOptSecondTokenB<SINGLE>>();
    Stage::Ptr fused_ops = make_stage<LoraFusedOps>();

    Stage::Ptr lora_hf_ref = make_stage<LoraRef<HORIZONTAL_FUSED>>();
    Stage::Ptr lora_opt_hf_first_token_a_ref = make_stage<LoraOptFirstTokenA<HORIZONTAL_FUSED>>("ref", 1, 1);
    Stage::Ptr lora_opt_hf_first_token_a_small = make_stage<LoraOptFirstTokenA<HORIZONTAL_FUSED>>("small", 8, 1);
    Stage::Ptr lora_opt_hf_first_token_a_medium = make_stage<LoraOptFirstTokenA<HORIZONTAL_FUSED>>("medium", 8, 2);
    Stage::Ptr lora_opt_hf_first_token_a_large = make_stage<LoraOptFirstTokenA<HORIZONTAL_FUSED>>("large", 16, 2);
    Stage::Ptr lora_opt_hf_first_token_b_ref = make_stage<LoraOptFirstTokenB<HORIZONTAL_FUSED>>("ref", 1, 1, 8);
    Stage::Ptr lora_opt_hf_first_token_b_small = make_stage<LoraOptFirstTokenB<HORIZONTAL_FUSED>>("small", 8, 1, 8);
    Stage::Ptr lora_opt_hf_first_token_b_medium = make_stage<LoraOptFirstTokenB<HORIZONTAL_FUSED>>("medium", 8, 2, 16);
    Stage::Ptr lora_opt_hf_first_token_b_large = make_stage<LoraOptFirstTokenB<HORIZONTAL_FUSED>>("large", 16, 2, 8);
    Stage::Ptr lora_opt_hf_second_token_a = make_stage<LoraOptSecondTokenA<HORIZONTAL_FUSED>>();
    Stage::Ptr lora_opt_hf_second_token_b = make_stage<LoraOptSecondTokenB<HORIZONTAL_FUSED>>();

    LoraImpl() : PrimitiveImplOCL(Lora::get_type_info_static()) {}
    LoraImpl(const program_node& node, const RuntimeParams& params) : LoraImpl() {
        size_t lora_count = LoraRefBase<>::get_lora_count(params);

        if (lora_count == 1) {
            add_stage(lora_ref, params);
            add_stage(lora_opt_first_token_a_ref, params);
            add_stage(lora_opt_first_token_a_small, params);
            add_stage(lora_opt_first_token_a_medium, params);
            add_stage(lora_opt_first_token_a_large, params);
            add_stage(lora_opt_first_token_b_ref, params);
            add_stage(lora_opt_first_token_b_medium, params);
            add_stage(lora_opt_first_token_b_large, params);
            add_stage(lora_opt_second_token_a, params);
            add_stage(lora_opt_second_token_b, params);
            add_stage(fused_ops, params);
        } else {
            add_stage(lora_hf_ref, params);
            add_stage(lora_opt_hf_first_token_a_ref, params);
            add_stage(lora_opt_hf_first_token_a_small, params);
            add_stage(lora_opt_hf_first_token_a_medium, params);
            add_stage(lora_opt_hf_first_token_a_large, params);
            add_stage(lora_opt_hf_first_token_b_ref, params);
            add_stage(lora_opt_hf_first_token_b_small, params);
            add_stage(lora_opt_hf_first_token_b_medium, params);
            add_stage(lora_opt_hf_first_token_b_large, params);
            add_stage(lora_opt_hf_second_token_a, params);
            add_stage(lora_opt_hf_second_token_b, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<LoraImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        auto desc = params.typed_desc<lora>();

        const auto& state_alpha_lo = params.input_layouts[3];
        size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);

        if (lora_rank == 0) {
            return {};
        }

        size_t lora_count = LoraRefBase<>::get_lora_count(params);

        if (LoraOptBase<>::is_first_token(params)) {
            const auto& lora_input_lo = params.input_layouts[1];
            size_t batch = extract_channel(ChannelName::BATCH, lora_input_lo);
            size_t feature = extract_channel(ChannelName::FEATURE, lora_input_lo);

            return {BufferDescriptor{lora_rank * batch * feature * lora_count, params.get_output_layout().data_type}};
        } else {
            const auto& state_a_lo = params.input_layouts[2];
            size_t input_state = extract_channel(ChannelName::FEATURE, state_a_lo);

            size_t max_workgroup_size = params.get_device_info().max_work_group_size;

            size_t gemma_sgK = max_workgroup_size / (lora_rank * lora_count);
            size_t gemma_wgs = ceil_div(input_state, LoraOptBase<>::gemm_a_sg_bk * gemma_sgK);

            return {BufferDescriptor{gemma_wgs * lora_count, params.get_output_layout().data_type}};
        }
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) const override {
        size_t is_single_lora = LoraRefBase<>::get_lora_count(*instance.get_impl_params()) == 1;
        if (is_single_lora) {
            return get_stages_execution_order_single_lora(instance);
        } else {
            return get_stages_execution_order_hf_lora(instance);
        }
    }
};

}  // namespace

std::unique_ptr<primitive_impl> Lora::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<lora>());
    return std::make_unique<LoraImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lora)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::LoraImpl)
