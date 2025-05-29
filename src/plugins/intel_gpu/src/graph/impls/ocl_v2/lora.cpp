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

enum KernelsTypes {
    REFERENCE = 0,
    FIRST_TOKEN_A_SMALL,
    FIRST_TOKEN_A_MEDIUM,
    FIRST_TOKEN_A_LARGE,
    FIRST_TOKEN_B_MEDIUM,
    FIRST_TOKEN_B_LARGE,
    SECOND_TOKEN_A,
    SECOND_TOKEN_B,
    FUSED_OPS
};

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

        jit_constants.make("LORA_COUNT", (desc->input_size() - 2ul) / 3ul);

        return jit_constants;
    }
};

class LoraRef : public LoraRefBase {
public:
    LoraRef() : LoraRefBase("lora_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraRefBase::get_jit_constants(params);

        jit_constants.make("BASE_KERNEL", 1);
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

class LoraOptBase : public LoraRefBase {
protected:
    explicit LoraOptBase(std::string_view suffix = "") : LoraRefBase("lora_opt", suffix) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraRefBase::get_jit_constants(params);
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
        auto in_dtype = params.get_input_layout().data_type;
        size_t max_gemma_sgk = in_dtype == ov::element::f16 ? 64ul : 32ul;
        return max_gemma_sgk;
    }

    static constexpr size_t gemm_a_sg_bk = 32ul;
};

class LoraOptFirstTokenBase : public LoraOptBase {
protected:
    explicit LoraOptFirstTokenBase(std::string_view suffix, size_t reg_m, size_t reg_n) : LoraOptBase(suffix), reg_m(reg_m), reg_n(reg_n) {}

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

    std::string generate_matmul_code(size_t M, size_t N, ov::element::Type_t dtype, bool is_a_kernel) const {
        std::string res = "";
        std::string int_type = dtype == ov::element::f16 ? "ushort" : "uint";
        std::string input_type = is_a_kernel ? "INPUT1_TYPE" : "ACCUMULATOR_TYPE";

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                res += "ACCUMULATOR_TYPE sum_" + std::to_string(m) + "_" + std::to_string(n) + " = 0;";
            }
        }

        res += "for (int i = 0; i < K; i += SUBGROUP_SIZE) {";

        for (size_t m = 0; m < M; ++m) {
            res += int_type + " input_" + std::to_string(m) + " = " + generate_block_read(dtype, "ptrA + " + std::to_string(m) + " * K") + ";";
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

        res += "ptrB += N; }";
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
                    res += generate_block_write(dtype,
                                                "ptrC + SUBGROUP_SIZE * " + std::to_string(n),
                                                "sum_" + std::to_string(m) + "_" + std::to_string(n) + " * alpha_" + std::to_string(n));
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
        auto jit_constants = LoraOptBase::get_jit_constants(params);

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

class LoraOptFirstTokenA : public LoraOptFirstTokenBase {
public:
    explicit LoraOptFirstTokenA(std::string suffix, size_t reg_m, size_t reg_n) : LoraOptFirstTokenBase("first_token_a_" + suffix, reg_m, reg_n) {}

    static std::pair<size_t, size_t> get_subgroup_params(const RuntimeParams& params, size_t reg_n) {
        size_t sg_m, sg_n;

        const auto& lora_input_lo = params.input_layouts[1];
        size_t lora_rank = extract_channel(ChannelName::FEATURE, lora_input_lo);

        size_t subgroup_size = get_subgroup_size(params);

        sg_n = lora_rank / subgroup_size / reg_n;
        if (sg_n != 0) {
            sg_m = ceil_div(subgroup_size, sg_n);
        }

        return {sg_m, sg_n};
    }

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptFirstTokenBase::get_jit_constants(params);

        jit_constants.make("FIRST_TOKEN_A", 1);

        jit_constants.make("N", "LORA_RANK");

        LayoutJitter state_a_dims(params.input_layouts[2], params.in_port_to_shape_info_offset.at(2));
        jit_constants.make("K", state_a_dims.dim(ChannelName::FEATURE));

        auto in_dtype = params.get_input_layout().data_type;
        jit_constants.make("MAIN_MATMUL_CODE", generate_matmul_code(reg_m, reg_n, in_dtype, true));

        jit_constants.make("MULTIPLY_AND_STORE_CODE", generate_store_result_code(reg_m, reg_n, in_dtype, true));

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

            size_t subgroup_size = LoraOptBase::get_subgroup_size(params);

            size_t sg_m, sg_n;
            std::tie(sg_m, sg_n) = LoraOptFirstTokenA::get_subgroup_params(params, reg_n);

            if (sg_n == 0) {
                return;
            }

            size_t bm = reg_m * sg_m;
            size_t bn = reg_n * sg_n * subgroup_size;

            wgs.global = {round_up_to(batch * feature, bm) / reg_m, round_up_to(lora_rank, bn) / reg_n, 1};
            wgs.local = {sg_m, sg_n * subgroup_size, 1};
        }};
    }
};

class LoraOptFirstTokenB : public LoraOptFirstTokenBase {
public:
    explicit LoraOptFirstTokenB(std::string suffix, size_t reg_m, size_t reg_n, size_t sg_m, size_t sg_n)
        : LoraOptFirstTokenBase("first_token_b_" + suffix, reg_m, reg_n),
          sg_m(sg_m),
          sg_n(sg_n) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptFirstTokenBase::get_jit_constants(params);

        jit_constants.make("FIRST_TOKEN_B", 1);

        LayoutJitter state_b_dims(params.input_layouts[4], params.in_port_to_shape_info_offset.at(4));
        jit_constants.make("N", state_b_dims.dim(ChannelName::BATCH));

        jit_constants.make("K", "LORA_RANK");

        auto in_dtype = params.get_input_layout().data_type;
        jit_constants.make("MAIN_MATMUL_CODE", generate_matmul_code(reg_m, reg_n, in_dtype, false));

        jit_constants.make("ADD_AND_STORE_CODE", generate_store_result_code(reg_m, reg_n, in_dtype, false));

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
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{
            [reg_m = reg_m, reg_n = reg_n, sg_m = sg_m, sg_n = sg_n](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
                assert(!params.is_dynamic());
                auto& wgs = kd.params.workGroups;

                const auto& lora_input_lo = params.input_layouts[1];
                size_t batch = extract_channel(ChannelName::BATCH, lora_input_lo);
                size_t feature = extract_channel(ChannelName::FEATURE, lora_input_lo);

                const auto& state_b_lo = params.input_layouts[4];
                size_t N = extract_channel(ChannelName::BATCH, state_b_lo);

                size_t subgroup_size = LoraOptBase::get_subgroup_size(params);

                size_t bm = reg_m * sg_m;
                size_t bn = reg_n * sg_n * subgroup_size;

                wgs.global = {round_up_to(batch * feature, bm) / reg_m, round_up_to(N, bn) / reg_n, 1};
                wgs.local = {sg_m, sg_n * subgroup_size, 1};
            }};
    }

    size_t sg_m;
    size_t sg_n;
};

class LoraOptSecondTokenA : public LoraOptBase {
public:
    LoraOptSecondTokenA() : LoraOptBase("second_token_a") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptBase::get_jit_constants(params);

        jit_constants.make("SECOND_TOKEN_A", 1);

        LayoutJitter state_a_dims(params.input_layouts[2], params.in_port_to_shape_info_offset.at(2));
        jit_constants.make("K", state_a_dims.dim(ChannelName::FEATURE));

        jit_constants.make("MAX_GEMMA_SGK", get_max_gemma_sgk(params));

        jit_constants.make("GEMMA_SGK", "min(MAX_WORKGROUP_SIZE / LORA_RANK, MAX_GEMMA_SGK)");

        jit_constants.make("GEMMA_SG_BK", gemm_a_sg_bk);

        jit_constants.make("MAX_GEMMA_SG_BK", 64);

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

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
            size_t gemma_sgK = max_workgroup_size / lora_rank;

            size_t K = extract_channel(ChannelName::FEATURE, params.input_layouts[2]);
            size_t gemma_wgs = ceil_div(K, LoraOptBase::gemm_a_sg_bk * gemma_sgK);

            wgs.global = {gemma_wgs, gemma_sgK, lora_rank};
            wgs.local = {1, gemma_sgK, lora_rank};
        }};
    }
};

class LoraOptSecondTokenB : public LoraOptBase {
public:
    LoraOptSecondTokenB() : LoraOptBase("second_token_b") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = LoraOptBase::get_jit_constants(params);

        jit_constants.make("SECOND_TOKEN_B", 1);

        LayoutJitter state_b_dims(params.input_layouts[4], params.in_port_to_shape_info_offset.at(4));
        jit_constants.make("N", state_b_dims.dim(ChannelName::BATCH));

        LayoutJitter state_a_dims(params.input_layouts[2], params.in_port_to_shape_info_offset.at(2));
        jit_constants.make("K", state_a_dims.dim(ChannelName::FEATURE));

        jit_constants.make("GEMMA_SG_BK", gemm_a_sg_bk);

        jit_constants.make("MAX_GEMMA_SGK", get_max_gemma_sgk(params));

        jit_constants.make("GEMMA_SGK", "min(MAX_WORKGROUP_SIZE / LORA_RANK, MAX_GEMMA_SGK)");

        jit_constants.make("GEMMB_PART_NUM", "CEIL_DIV(K, GEMMA_SG_BK * GEMMA_SGK)");

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            size_t max_workgroup_size = params.get_device_info().max_work_group_size;
            size_t output_state = extract_channel(ChannelName::BATCH, params.input_layouts[4]);

            wgs.global = {round_up_to(output_state, max_workgroup_size), 1, 1};
            wgs.local = {max_workgroup_size, 1, 1};
        }};
    }
};

bool is_optimized_kernel_supported(const cldnn::primitive_inst& instance) {
    size_t subgroup_size = LoraOptBase::get_subgroup_size(*instance.get_impl_params());

    const auto& state_a_layout = instance.get_input_layout(2);
    size_t input_state = state_a_layout.get_shape().back();
    if (input_state % subgroup_size != 0) {
        return false;
    }

    const auto& alpha_layout = instance.get_input_layout(3);
    size_t lora_rank = alpha_layout.get_shape().back();
    if (lora_rank % subgroup_size != 0) {
        return false;
    }

    const auto& state_b_layout = instance.get_input_layout(4);
    size_t output_state = state_b_layout.get_shape().front();
    if (output_state % subgroup_size != 0) {
        return false;
    }

    return true;
}

std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) {
    std::vector<size_t> stages_order;

    bool is_empty_lora = instance.get_input_layout(2).count() == 0;
    if (!is_empty_lora) {
        if (!is_optimized_kernel_supported(instance)) {
            stages_order.emplace_back(KernelsTypes::REFERENCE);
        } else {
            const auto& lora_input = instance.get_input_layout(1);
            size_t batch = extract_channel(ChannelName::BATCH, lora_input);
            size_t feature = extract_channel(ChannelName::FEATURE, lora_input);
            bool is_first_token = batch * feature > 1;

            if (is_first_token) {
                const auto& state_alpha_lo = instance.get_input_layout(3);
                size_t lora_rank = extract_channel(ChannelName::FEATURE, state_alpha_lo);

                if (lora_rank == 128 || lora_rank == 256) {
                    stages_order.emplace_back(KernelsTypes::FIRST_TOKEN_A_LARGE);
                } else if (lora_rank == 64) {
                    stages_order.emplace_back(KernelsTypes::FIRST_TOKEN_A_MEDIUM);
                } else {
                    stages_order.emplace_back(KernelsTypes::FIRST_TOKEN_A_SMALL);
                }

                if (batch * feature < 256) {
                    size_t max_workgroup_size = instance.get_impl_params()->get_device_info().max_work_group_size;
                    size_t subgroup_size = LoraOptBase::get_subgroup_size(*instance.get_impl_params());
                    size_t b_medium_sg_m = 16;
                    size_t b_medium_sg_n = 4;
                    if (b_medium_sg_m * b_medium_sg_n * subgroup_size > max_workgroup_size) {
                        stages_order.emplace_back(KernelsTypes::FIRST_TOKEN_B_LARGE);
                    } else {
                        stages_order.emplace_back(KernelsTypes::FIRST_TOKEN_B_MEDIUM);
                    }
                } else {
                    stages_order.emplace_back(KernelsTypes::FIRST_TOKEN_B_LARGE);
                }
            } else {
                stages_order.emplace_back(KernelsTypes::SECOND_TOKEN_A);
                stages_order.emplace_back(KernelsTypes::SECOND_TOKEN_B);
            }
        }
    }

    if (instance.has_fused_primitives()) {
        stages_order.emplace_back(KernelsTypes::FUSED_OPS);
    }

    return stages_order;
}

class LoraImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::LoraImpl)

    Stage::Ptr lora_ref = make_stage<LoraRef>();
    Stage::Ptr lora_opt_first_token_a_small = make_stage<LoraOptFirstTokenA>("small", 4, 1);
    Stage::Ptr lora_opt_first_token_a_medium = make_stage<LoraOptFirstTokenA>("medium", 8, 2);
    Stage::Ptr lora_opt_first_token_a_large = make_stage<LoraOptFirstTokenA>("large", 16, 2);
    Stage::Ptr lora_opt_first_token_b_medium = make_stage<LoraOptFirstTokenB>("medium", 8, 2, 16, 4);
    Stage::Ptr lora_opt_first_token_b_large = make_stage<LoraOptFirstTokenB>("large", 16, 2, 8, 4);
    Stage::Ptr lora_opt_second_token_a = make_stage<LoraOptSecondTokenA>();
    Stage::Ptr lora_opt_second_token_b = make_stage<LoraOptSecondTokenB>();
    Stage::Ptr fused_ops = make_stage<LoraFusedOps>();

    LoraImpl() : PrimitiveImplOCL(Lora::get_type_info_static()) {}
    LoraImpl(const program_node& node, const RuntimeParams& params) : LoraImpl() {
        add_stage(lora_ref, params);
        add_stage(lora_opt_first_token_a_small, params);
        add_stage(lora_opt_first_token_a_medium, params);
        add_stage(lora_opt_first_token_a_large, params);
        add_stage(lora_opt_first_token_b_medium, params);
        add_stage(lora_opt_first_token_b_large, params);
        add_stage(lora_opt_second_token_a, params);
        add_stage(lora_opt_second_token_b, params);
        add_stage(fused_ops, params);
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

        const auto& lora_input_lo = params.input_layouts[1];
        size_t batch = extract_channel(ChannelName::BATCH, lora_input_lo);
        size_t feature = extract_channel(ChannelName::FEATURE, lora_input_lo);

        auto first_token_buffer = BufferDescriptor{lora_rank * batch * feature, params.get_output_layout().data_type};

        const auto& state_a_lo = params.input_layouts[2];
        size_t input_state = extract_channel(ChannelName::FEATURE, state_a_lo);

        size_t max_workgroup_size = params.get_device_info().max_work_group_size;
        size_t output_a_size = ceil_div(input_state, LoraOptBase::gemm_a_sg_bk * (max_workgroup_size / lora_rank));

        auto second_token_buffer = BufferDescriptor{output_a_size, params.get_output_layout().data_type};

        return {first_token_buffer, second_token_buffer};
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        cldnn::stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }

        update_rt_params(instance);

        std::vector<cldnn::event::ptr> tmp_events(events);
        const auto& exec_stages = get_stages_execution_order(instance);
        for (const auto& stage_id : exec_stages) {
            tmp_events = {execute_stage(tmp_events, instance, *_stages[stage_id])};
        }

        return tmp_events[0];
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
