// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/constant_folding.hpp>
#include "fc_bias_fusion.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/manager.hpp"
#include "reshape_fc_fusion.hpp"
#include "align_matmul_input_ranks.hpp"
#include "convert_broadcast_to_tiles.hpp"
#include "convert_tile_to_seq_tiles.hpp"
#include "convert_matmul_to_fc.hpp"
#include "convert_to_power_static.hpp"
#include "convert_to_leaky_relu.hpp"
#include "convert_to_swish_cpu.hpp"
#include "rnn_sequences_optimization.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "ngram_fusion.hpp"

#include "itt.hpp"

namespace ov {
namespace intel_cpu {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ov::Model> &model) {
    RUN_ON_FUNCTION_SCOPE(ConvertToCPUSpecificOpset);

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ConvertMatMulToFC>();
    manager.register_pass<AlignMatMulInputRanks>();
    manager.register_pass<ConvertTileToSeqTiles>();
    manager.register_pass<FullyConnectedBiasFusion>();
    manager.register_pass<ConvertToPowerStatic>();
    manager.register_pass<ConvertToLeakyRelu>();
    manager.register_pass<ConvertToSwishCPU>();
    manager.register_pass<OptimizeSequenceTransposes>();
    if (!ov::op::util::has_op_with_type<ov::op::v0::FakeQuantize>(model)) {
        manager.register_pass<ReshapeFullyConnectedFusion>();
    }
    // after transformation "MoveEltwiseUpThroughDataMov" there can be Reshape sequences that should be eliminated or fused
    manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<NgramFusion>();
    manager.register_pass<ov::pass::Validate>();

    manager.run_passes(model);
}

}   // namespace intel_cpu
}   // namespace ov
