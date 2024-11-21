// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "common/pass/align_matmul_input_ranks.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "common/pass/convert_tile_to_seq_tiles.hpp"
#include "common/pass/convert_matmul_to_fc.hpp"
#include "common/pass/convert_to_power_static.hpp"
#include "common/pass/convert_to_leaky_relu.hpp"
#include "common/pass/convert_to_swish_cpu.hpp"
#include "common/pass/move_fc_reshape_to_weights.hpp"
#include "common/pass/fc_bias_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/op_conversions/convert_fc_to_quantized_legacy.hpp"
#include "common/pass/rnn_sequences_optimization.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/defs.hpp"
#include "config.h"

#if defined(OPENVINO_ARCH_X86_64)
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

#include "itt.hpp"

namespace ov {
namespace intel_cpu {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ov::Model> &model, const Config& config) {
    RUN_ON_FUNCTION_SCOPE(ConvertToCPUSpecificOpset);

    ov::pass::Manager manager("CPU:ConvertToCPUSpecificOpset");
    manager.set_per_pass_validation(false);

    // CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertMatMulToFC);
    if (std::getenv("EXTRA_DUMP")) {
        manager.run_passes(model);
        ov::pass::Serialize("after_fc.xml", "/dev/null").run_on_model(model);
        CPU_DISABLE_PASS_COMMON(manager, ConvertMatMulToFC);
    }

    CPU_REGISTER_PASS_COMMON(manager, FullyConnectedBiasFusion);

    std::vector<ov::element::Type> supported_activation_types {
        // @todo enable for bf16 as well
        // after EnforceInferencePrecision is replaced with ConvertPrecision
        ov::element::f32,
    };

    std::vector<ov::element::Type> supported_compressed_weights_types {
        ov::element::u8,
        ov::element::i8,
        ov::element::u4,
        ov::element::i4,
        ov::element::nf4,
        ov::element::f4e2m1,
    };

#if defined(OPENVINO_ARCH_X86_64)
    // @todo introduce something like CPU_REGISTER_PASS_X64_AVX2
    const bool isDecompressionSupported = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2);
    if (isDecompressionSupported) {
        CPU_REGISTER_PASS_X64(
            manager,
            pass::ConvertFullyConnectedToFullyConnectedCompressed,
            supported_activation_types,
            supported_compressed_weights_types,
            [&config](const std::shared_ptr<ov::op::internal::FullyConnected>& fc, size_t IC, size_t OC, size_t G) {
                // @todo replace 'inferencePrecision' check with 'fc->get_input_element_type(0) == ov::element::bf16'
                // after bf16 pipeline is moved to ConvertPrecision
                if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
                    config.inferencePrecision == ov::element::bf16) {
                    // OneDNN AMX IP implementation has limited shapes support due to performance considerations. As a
                    // current solution conditions below are copied from OneDNN to make sure correct IP impl will be
                    // used since fallback one doesn't support weights decompression feature.
                    size_t simdWidth = 16;
                    size_t vnniFactor = 2;
                    size_t maxSize = 512;
                    auto amxRow = vnniFactor * simdWidth;

                    if ((IC <= amxRow && OC <= amxRow) || (IC <= maxSize && OC <= maxSize && IC % amxRow != 0)) {
                        return false;
                    }
                }

                if (IC % G != 0 || IC / G < 4 || OC == 1) {
                    return false;
                }
                return true;
            });
    }
#endif // OPENVINO_ARCH_X86_64

    CPU_REGISTER_PASS_X64(manager, pass::ConvertFCToFCQuantizedLegacy);
    if (std::getenv("EXTRA_DUMP")) {
        manager.run_passes(model);
        ov::pass::Serialize("after_fc_quantized.xml", "/dev/null").run_on_model(model);
        CPU_DISABLE_PASS_COMMON(manager, ConvertMatMulToFC);
    }

    CPU_REGISTER_PASS_X64(manager, MoveFCReshapeToWeights);
    CPU_REGISTER_PASS_X64(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertTileToSeqTiles);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToPowerStatic);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToLeakyRelu);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToSwishCPU);
    CPU_REGISTER_PASS_COMMON(manager, OptimizeSequenceTransposes);
    // after transformation "MoveEltwiseUpThroughDataMov" there can be reshaped sequences that should be eliminated or fused
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ReshapeSequenceFusion);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager,
                             ov::pass::ConvertPrecision,
                             precisions_map{{ov::element::i64, ov::element::i32}},
                             type_to_fuse_map{{}},
                             false,
                             false);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert); // Need to clean up after the ConvertPrecision.

    manager.run_passes(model);
}

}   // namespace intel_cpu
}   // namespace ov
