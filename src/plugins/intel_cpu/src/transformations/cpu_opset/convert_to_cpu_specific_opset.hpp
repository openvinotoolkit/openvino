// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pass/align_matmul_input_ranks.hpp"
#include "common/pass/convert_matmul_to_fc.hpp"
#include "common/pass/convert_tile_to_seq_tiles.hpp"
#include "common/pass/convert_to_leaky_relu.hpp"
#include "common/pass/convert_to_power_static.hpp"
#include "common/pass/convert_to_swish_cpu.hpp"
#include "common/pass/fc_bias_fusion.hpp"
#include "common/pass/move_fc_reshape_to_weights.hpp"
#include "common/pass/move_readvalue_inputs_to_subgraph.hpp"
#include "common/pass/rnn_sequences_optimization.hpp"
#include "config.h"
#include "nodes/fullyconnected.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/defs.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/op_conversions/convert_fc_to_quantized_legacy.hpp"

namespace ov::intel_cpu {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ov::Model>& model, const Config& config) {
    RUN_ON_FUNCTION_SCOPE(ConvertToCPUSpecificOpset);

    ov::pass::Manager manager("CPU:ConvertToCPUSpecificOpset");
    manager.set_per_pass_validation(false);

    CPU_REGISTER_PASS_COMMON(manager, ConvertMatMulToFC);
    CPU_REGISTER_PASS_COMMON(manager, FullyConnectedBiasFusion);

    CPU_REGISTER_PASS_COMMON(
        manager,
        pass::ConvertFullyConnectedToFullyConnectedCompressed,
        ov::intel_cpu::node::FullyConnected::getSupportedCompressedActivationsTypes(),
        ov::intel_cpu::node::FullyConnected::getSupportedCompressedWeightsTypes(),
        [&config](const std::shared_ptr<ov::op::internal::FullyConnected>& fc, size_t IC, size_t OC, size_t G) {
            return ov::intel_cpu::node::FullyConnected::isSupportedCompressedOperation(fc,
                                                                                       IC,
                                                                                       OC,
                                                                                       G,
                                                                                       config.inferencePrecision);
        });

    CPU_REGISTER_PASS_X64(manager, pass::ConvertFCToFCQuantizedLegacy);
    CPU_REGISTER_PASS_COMMON(manager, MoveFCReshapeToWeights);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertTileToSeqTiles);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToPowerStatic);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToLeakyRelu);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToSwishCPU);
    CPU_REGISTER_PASS_COMMON(manager, OptimizeSequenceTransposes);
    // after transformation "MoveEltwiseUpThroughDataMov" there can be reshaped sequences that should be eliminated or
    // fused
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ReshapeSequenceFusion);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager,
                             ov::pass::ConvertPrecision,
                             precisions_map{{ov::element::i64, ov::element::i32}},
                             type_to_fuse_map{{}},
                             false,
                             false);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert);  // Need to clean up after the ConvertPrecision.
    CPU_REGISTER_PASS_COMMON(manager, MoveReadValueInputsToSubgraph);

    manager.run_passes(model);
}

}  // namespace ov::intel_cpu
