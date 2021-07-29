// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/constant_folding.hpp>
#include "convert_matmul_to_fc_or_gemm.hpp"
#include "fc_bias_fusion.hpp"
#include "reshape_fc_fusion.hpp"
#include "reshape_fully_connected.hpp"
#include "convert_broadcast_to_tiles.hpp"
#include "convert_tile_to_seq_tiles.hpp"
#include "reshape_1d_ops.hpp"
#include "convert_to_power_static.hpp"
#include "convert_to_leaky_relu.hpp"
#include "convert_to_swish_cpu.hpp"
#include "reshape_prelu.hpp"
#include "rnn_sequences_optimization.hpp"

namespace MKLDNNPlugin {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ngraph::Function> &nGraphFunc) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<Reshape1DConvolution>();
    manager.register_pass<Reshape1DGroupConvolution>();
    manager.register_pass<Reshape1DAvgPool>();
    manager.register_pass<Reshape1DMaxPool>();
    manager.register_pass<ConvertBroadcastToTiles>();
    manager.register_pass<ConvertTileToSeqTiles>();
    manager.register_pass<ConvertMatMulToFC>();
    manager.register_pass<ConvertMatMulToGemm>();
    manager.register_pass<FullyConnectedBiasFusion>();
    manager.register_pass<ReshapeFullyConnected>();
    manager.register_pass<ConvertToPowerStatic>();
    manager.register_pass<ConvertToLeakyRelu>();
    manager.register_pass<ReshapePRelu>();
    manager.register_pass<ConvertToSwishCPU>();
    manager.register_pass<OptimizeGRUSequenceTransposes>();
    manager.register_pass<OptimizeLSTMSequenceTransposes>();
    manager.register_pass<OptimizeRNNSequenceTransposes>();
    if (!ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc)) {
        manager.register_pass<ReshapeFullyConnectedFusion>();
    }
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ ngraph::element::i64, ngraph::element::i32 }});
    manager.run_passes(nGraphFunc);
}

}  // namespace MKLDNNPlugin