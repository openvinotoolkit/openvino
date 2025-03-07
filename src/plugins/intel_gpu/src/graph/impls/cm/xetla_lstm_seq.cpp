// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xetla_lstm_seq.hpp"

#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_lstm_build_options() {
    return " -Qxcm_jit_option=-DPASTokenReduction "
           " -mllvm --vc-disable-indvars-opt=true "
           " /Qxcm_jit_option=-enableBCR /Qxcm_doubleGRF "
           " -DXETLA_CODE_BASE=__CM__ ";
}
class XetlaLSTMLoopGenerator : public KernelGenerator {
public:
    XetlaLSTMLoopGenerator() : KernelGenerator("xetla_lstm_loop") {}

protected:
    std::string get_build_options(const kernel_impl_params& params) const override {
        return KernelGenerator::get_build_options(params) + get_lstm_build_options();
    }

    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<lstm_seq>();
        const auto& x_shape = params.input_layouts[0].get_shape();

        jit_constants.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("INPUT_SIZE", x_shape[2]),
        });

        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});
        args.push_back({ArgumentDescriptor::Types::INPUT, 6});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 2});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& x_shape = params.input_layouts[0].get_shape();
            const auto& ini_hidden_shape = params.input_layouts[1].get_shape();
            const auto& out_shape = params.output_layouts[0].get_shape();

            const auto num_gates = 4;
            const auto hidden_size = ini_hidden_shape[2];
            const auto num_dir = ini_hidden_shape[1];

            size_t wg_m_hh = 1;
            size_t wg_n_hh = hidden_size * num_gates;

            size_t sg_m_hh = 1;
            size_t sg_n_hh = 16;

            size_t matrix_m_hh = 1;
            size_t matrix_n_hh = hidden_size * num_gates;

            size_t group_range_m = (matrix_m_hh + wg_m_hh - 1) / wg_m_hh;
            size_t group_range_n = (matrix_n_hh + wg_n_hh - 1) / wg_n_hh;
            size_t subgroup_range_m = (wg_m_hh + sg_m_hh - 1) / sg_m_hh;
            size_t subgroup_range_n = (wg_n_hh + sg_n_hh - 1) / sg_n_hh;

            wgs.global = {group_range_n * subgroup_range_n, group_range_m * subgroup_range_m, num_dir};
            wgs.local = {subgroup_range_n, subgroup_range_m, 1};
        };

        return f;
    }
};

class XetlaLSTMGemmGenerator : public KernelGenerator {
public:
    XetlaLSTMGemmGenerator() : KernelGenerator("xetla_lstm_gemm") {}

protected:
    std::string get_build_options(const kernel_impl_params& params) const override {
        return KernelGenerator::get_build_options(params) + get_lstm_build_options();
    }

    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<lstm_seq>();
        const auto& x_shape = params.input_layouts[0].get_shape();

        jit_constants.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("INPUT_SIZE", x_shape[2]),
        });

        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        args.push_back({ArgumentDescriptor::Types::INPUT, 5});
        args.push_back({ArgumentDescriptor::Types::INPUT, 6});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& x_shape = params.input_layouts[0].get_shape();
            const auto& ini_hidden_shape = params.input_layouts[1].get_shape();
            const auto& out_shape = params.output_layouts[0].get_shape();

            const auto num_gates = 4;
            const auto hidden_size = ini_hidden_shape[2];
            const auto seq_len = out_shape[2];
            const auto num_dir = ini_hidden_shape[1];

            size_t matrix_m_ih = seq_len;
            size_t matrix_n_ih = hidden_size * num_gates;

            size_t wg_m_ih = 40;
            size_t wg_n_ih = 256;

            size_t sg_m_ih = 24;
            size_t sg_n_ih = 32;

            size_t local_kslicing_ih = 1;
            size_t subgroup_range_m = (wg_m_ih + sg_m_ih - 1) / sg_m_ih;
            size_t subgroup_range_n = (wg_n_ih + sg_n_ih - 1) / sg_n_ih;

            size_t group_range_m = (matrix_m_ih + wg_m_ih - 1) / wg_m_ih;
            size_t group_range_n = (matrix_n_ih + wg_n_ih - 1) / wg_n_ih;

            wgs.global = {group_range_n * subgroup_range_n, group_range_m * subgroup_range_m, num_dir * local_kslicing_ih};
            wgs.local = {subgroup_range_n, subgroup_range_m, local_kslicing_ih};
        };

        return f;
    }
};

class LSTMImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::LSTMImpl)
    Stage::Ptr lstm_loop = make_stage<XetlaLSTMLoopGenerator>();
    Stage::Ptr lstm_gemm = make_stage<XetlaLSTMGemmGenerator>();

    LSTMImpl() : PrimitiveImplOCL(LSTMSeqImplementationManager::get_type_info_static()) {}
    LSTMImpl(const program_node& node, const kernel_impl_params& params) : LSTMImpl() {
        add_stage(lstm_loop, params);
        add_stage(lstm_gemm, params);
    }
    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<LSTMImpl>(this);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto x_shape = params.input_layouts[0].get_shape();
        const auto ini_hidden_shape = params.input_layouts[1].get_shape();
        const auto out_shape = params.output_layouts[0].get_shape();

        const auto num_gates = 4;
        const auto batch_size = x_shape[0];
        const auto hidden_size = ini_hidden_shape[2];
        const auto seq_len = out_shape[2];
        const auto num_dir = ini_hidden_shape[1];

        auto buf_size = num_dir * seq_len * batch_size * hidden_size * num_gates;
        return {BufferDescriptor{buf_size, ov::element::f32}};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> LSTMSeqImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<lstm_seq>());
    return std::make_unique<LSTMImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::LSTMImpl)
