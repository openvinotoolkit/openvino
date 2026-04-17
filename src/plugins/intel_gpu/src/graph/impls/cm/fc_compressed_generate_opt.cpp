// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_compressed_generate_opt.hpp"

#include "common_utils/kernel_generator_base.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

static constexpr int VEC_SIZE = 8;
static constexpr int WG_SIZE  = 64;

// -----------------------------------------------------------------------
// Helper: detect weight ZP and activation scale indices.
// Same logic as ocl::FCCompressedGenerateOpt.
// -----------------------------------------------------------------------
static bool detect_has_zp(const kernel_impl_params& params) {
    if (params.input_layouts.size() <= 3)
        return false;
    const auto dt = params.input_layouts[3].data_type;
    return dt == data_types::u4 || dt == data_types::i4 || dt == data_types::u8;
}

static bool detect_zp_is_u8(const kernel_impl_params& params) {
    return detect_has_zp(params) &&
           params.input_layouts[3].data_type == data_types::u8;
}

static size_t act_scale_index(const kernel_impl_params& params) {
    const bool act_is_i8 = (params.input_layouts[0].data_type == data_types::i8);
    if (!act_is_i8)
        return SIZE_MAX;
    const size_t candidate = detect_has_zp(params) ? 4 : 3;
    if (candidate >= params.input_layouts.size())
        return SIZE_MAX;
    const auto& lay = params.input_layouts[candidate];
    if (lay.data_type != data_types::f16)
        return SIZE_MAX;
    return candidate;
}

// -----------------------------------------------------------------------
// CM Kernel Generator for fc_compressed_generate_opt.cm
// -----------------------------------------------------------------------
class FCCompressedOptCMGenerator : public KernelGenerator {
public:
    FCCompressedOptCMGenerator() : KernelGenerator("fc_compressed_generate_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto& in0   = params.input_layouts[0];  // activation
        const auto& in1   = params.input_layouts[1];  // weight
        const auto& in_sc = params.input_layouts[2];  // weight scale

        const auto& shape_a  = in0.get_shape();
        const auto& shape_w  = in1.get_shape();
        const auto& shape_sc = in_sc.get_shape();

        const size_t rank = shape_a.size();
        const size_t K    = shape_a[rank - 1];
        const size_t N    = shape_w[0];
        size_t B = 1;
        for (size_t i = 0; i + 1 < rank; ++i)
            B *= shape_a[i];

        const size_t actual_num_groups = shape_sc[shape_sc.size() - 1];
        const size_t actual_group_size = (actual_num_groups > 0) ? (K / actual_num_groups) : K;

        const bool weight_is_signed = (in1.data_type == data_types::i4);
        const bool has_zp = detect_has_zp(params);
        const bool act_is_i8 = (in0.data_type == data_types::i8);
        const bool zp_is_u8 = detect_zp_is_u8(params);

        // Add KERNEL_NAME for CM entry point.
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        jit.add({
            make_jit_constant("K_SIZE",           static_cast<int>(K)),
            make_jit_constant("N_SIZE",           static_cast<int>(N)),
            make_jit_constant("B_SIZE",           static_cast<int>(B)),
            make_jit_constant("VEC_SIZE",         VEC_SIZE),
            make_jit_constant("WG_SIZE",          WG_SIZE),
        });

        jit.add({
            make_jit_constant("IS_WEIGHT_INT4",   1),
            make_jit_constant("IS_ACT_INT8",      act_is_i8 ? 1 : 0),
            make_jit_constant("WEIGHT_IS_SIGNED", weight_is_signed ? 1 : 0),
            make_jit_constant("HAS_ZP",           has_zp ? 1 : 0),
            make_jit_constant("ZP_IS_U8",         zp_is_u8 ? 1 : 0),
            make_jit_constant("GROUP_SIZE",       static_cast<int>(actual_group_size)),
            make_jit_constant("NUM_GROUPS",       static_cast<int>(actual_num_groups)),
        });

        return jit;
    }

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override {
        // -cmc is mandatory for CM compilation.
        return KernelGenerator::get_build_options(params);
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        // arg[0]: activation (f16 or i8)
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        // arg[1]: weight (u4/i4 packed)
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // arg[2]: weight scale (f16)
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});

        // arg[3]: weight ZP (u4, optional)
        const bool has_zp = detect_has_zp(params);
        if (has_zp) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        }

        // For W4A8: per-token activation scale (f16)
        const size_t as_idx = act_scale_index(params);
        if (as_idx != SIZE_MAX) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(as_idx)});
        }

        // output
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const kernel_impl_params& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());

            const auto& in1   = params.input_layouts[1];
            const auto& in0   = params.input_layouts[0];
            const auto& shape_a = in0.get_shape();

            const size_t rank  = shape_a.size();
            const size_t N     = in1.get_shape()[0];
            size_t B = 1;
            for (size_t i = 0; i + 1 < rank; ++i)
                B *= shape_a[i];

            const size_t N_padded = ((N + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
            auto& wgs = kd.params.workGroups;
            wgs.global = {N_padded, B, 1};
            wgs.local  = {static_cast<size_t>(WG_SIZE), 1, 1};
        }};
    }
};

// -----------------------------------------------------------------------
// Concrete CM implementation class
// -----------------------------------------------------------------------
class FCCompressedOptCMImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::FCCompressedOptCMImpl)

    Stage::Ptr gemv_stage = make_stage<FCCompressedOptCMGenerator>();

    FCCompressedOptCMImpl() : PrimitiveImplCM(FCCompressedGenerateOptCM::get_type_info_static()) {}
    explicit FCCompressedOptCMImpl(const kernel_impl_params& params) : FCCompressedOptCMImpl() {
        add_stage(gemv_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FCCompressedOptCMImpl>(this);
    }

    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data data;
        const auto& params = *instance.get_impl_params();
        const size_t n = params.input_layouts.size();
        const auto* fc_inst = dynamic_cast<const cldnn::fully_connected_inst*>(&instance);
        for (size_t i = 0; i < n; ++i) {
            if (i == 1 && fc_inst) {
                data.inputs.push_back(fc_inst->weights_memory());
            } else {
                data.inputs.push_back(instance.dep_memory_ptr(i));
            }
        }
        for (size_t i = 0; i < instance.outputs_memory_count(); ++i)
            data.outputs.push_back(instance.output_memory_ptr(i));
        if (instance.has_fused_primitives()) {
            for (size_t i = 0; i < instance.get_fused_mem_count(); ++i)
                data.fused_op_inputs.push_back(instance.fused_memory(i));
        }
        data.shape_info = instance.shape_info_memory_ptr();

        return data;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FCCompressedGenerateOptCM::create_impl(const program_node& node,
                                                                        const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<FCCompressedOptCMImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::FCCompressedOptCMImpl)
