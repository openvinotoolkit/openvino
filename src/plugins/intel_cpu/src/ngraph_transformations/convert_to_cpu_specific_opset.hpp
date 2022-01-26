// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/constant_folding.hpp>
#include "fc_bias_fusion.hpp"
#include "ngraph/op/fake_quantize.hpp"
#include "ngraph/pass/manager.hpp"
#include "reshape_fc_fusion.hpp"
#include "align_matmul_input_ranks.hpp"
#include "reshape_prelu.hpp"
#include "convert_broadcast_to_tiles.hpp"
#include "convert_tile_to_seq_tiles.hpp"
#include "convert_matmul_to_fc.hpp"
#include "convert_to_power_static.hpp"
#include "convert_to_leaky_relu.hpp"
#include "convert_to_swish_cpu.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/utils/utils.hpp"
#include "rnn_sequences_optimization.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "affinity_switcher.hpp"
#include "switch_affinity.hpp"

#include "markup_optimal_bs.hpp"
#include "form_components_with_unified_batch.hpp"

#include <transformations/serialize.hpp>

namespace ov {
namespace intel_cpu {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ngraph::Function> &nGraphFunc) {
    ngraph::pass::Manager manager;
    manager.register_pass<ConvertMatMulToFC>();
    manager.register_pass<AlignMatMulInputRanks>();
    manager.register_pass<ConvertTileToSeqTiles>();
    manager.register_pass<FullyConnectedBiasFusion>();
    manager.register_pass<ConvertToPowerStatic>();
    manager.register_pass<ConvertToLeakyRelu>();
    manager.register_pass<ConvertToSwishCPU>();
    manager.register_pass<OptimizeSequenceTransposes>();
    if (!ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc)) {
        manager.register_pass<ReshapeFullyConnectedFusion>();
    }
    // after transformation "MoveEltwiseUpThroughDataMov" there can be Reshape sequences that should be eliminated or fused
    manager.register_pass<ngraph::pass::ReshapeSequenceFusion>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ ngraph::element::i64, ngraph::element::i32 }});
    manager.register_pass<ngraph::pass::Serialize>("C://models//test.xml", "C://models//test.bin");
    manager.register_pass<MarkupOptimalBS>();
    manager.register_pass<FormComponentsWithUnifiedBatch>();
    manager.register_pass<ngraph::pass::VisualizeTree>("C://models//model//test.before");
    // TODO: remove 'share_constants' parameter
    manager.register_pass<SwitchAffinity>(false);
    manager.register_pass<ngraph::pass::Serialize>("C://models//affinity.xml",
                                                   "C://models//affinity.bin");
    manager.register_pass<ngraph::pass::VisualizeTree>("C://models//model//test.after");
    manager.get_pass_config()->set_callback<SwitchAffinity>([](const std::shared_ptr<const ov::Node>& n) -> bool {
        return n->get_friendly_name() == "resnet_model/conv2d/Conv2D";
    });

    manager.run_passes(nGraphFunc);
}

}   // namespace intel_cpu
}   // namespace ov
