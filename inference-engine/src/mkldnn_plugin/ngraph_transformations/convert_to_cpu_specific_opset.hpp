// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/constant_folding.hpp>
#include "fc_bias_fusion.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/fake_quantize.hpp"
#include "ngraph/pass/manager.hpp"
#include "reshape_fc_fusion.hpp"
#include "reshape_fully_connected.hpp"
#include "align_matmul_input_ranks.hpp"
#include "reshape_prelu.hpp"
#include "convert_broadcast_to_tiles.hpp"
#include "convert_tile_to_seq_tiles.hpp"
#include "convert_matmul_to_fc.hpp"
#include "convert_to_power_static.hpp"
#include "convert_to_leaky_relu.hpp"
#include "convert_to_swish_cpu.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/op_conversions/fq_decomposition.hpp"
#include "transformations/utils/utils.hpp"
#include "rnn_sequences_optimization.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "op/fully_connected.hpp"
#include "utils/general_utils.h"

namespace MKLDNNPlugin {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ngraph::Function> &nGraphFunc) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ConvertMatMulToFC>();
    manager.register_pass<AlignMatMulInputRanks>();
    manager.register_pass<ConvertTileToSeqTiles>();
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

    // TODO: remove after dynamic shapes support in FullyConnectedNode
    manager.get_pass_config()->set_callback<ConvertMatMulToFC>([](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        return node->get_input_partial_shape(0).is_dynamic();
    });

    auto isBinarizationFQ = [](const std::shared_ptr<const ngraph::Node> &node) -> bool {
        if (const auto& fq = std::dynamic_pointer_cast<const ngraph::op::v0::FakeQuantize>(node))
            if (fq->get_levels() == 2)
                return true;

        return false;
    };

    auto FQMostLikelyToBeFused = [](const std::shared_ptr<const ngraph::Node> &node) -> bool {
        // parent has single output
        if (node->get_input_node_ptr(0)->output(0).get_target_inputs().size() != 1)
            return false;

        // parent supports FQ fusing
        if (!one_of(node->get_input_node_ptr(0)->get_type_info(),
                    // Convolution
                    ngraph::op::v1::Convolution::get_type_info_static(),
                    ngraph::op::v1::GroupConvolution::get_type_info_static(),
                    // Deconvolution
                    ngraph::op::v1::ConvolutionBackpropData::get_type_info_static(),
                    // Pooling
                    ngraph::op::v1::AvgPool::get_type_info_static(),
                    // MatMul
                    ngraph::op::v0::MatMul::get_type_info_static(),
                    // FullyConnected
                    MKLDNNPlugin::FullyConnectedNode ::get_type_info_static(),
                    // Interpolate
                    ngraph::op::v4::Interpolate::get_type_info_static(),
                    // MVN
                    ngraph::op::v0::MVN::get_type_info_static(),
                    // Normalize
                    ngraph::op::v0::NormalizeL2::get_type_info_static(),
                    // Reduce
                    ngraph::op::util::ArithmeticReductionKeepDims::get_type_info_static(),
                    ngraph::op::util::LogicalReductionKeepDims::get_type_info_static(),
                    // Eltwise
                    ngraph::op::v1::Add::get_type_info_static(),
                    ngraph::op::v1::Subtract::get_type_info_static(),
                    ngraph::op::v1::Multiply::get_type_info_static(),
                    ngraph::op::v1::Divide::get_type_info_static(),
                    ngraph::op::v0::SquaredDifference::get_type_info_static(),
                    ngraph::op::v1::Maximum::get_type_info_static(),
                    ngraph::op::v1::Minimum::get_type_info_static(),
                    ngraph::op::v1::Mod::get_type_info_static(),
                    ngraph::op::v1::FloorMod::get_type_info_static(),
                    ngraph::op::v1::Power::get_type_info_static(),
                    ngraph::op::v1::Equal::get_type_info_static(),
                    ngraph::op::v1::NotEqual::get_type_info_static(),
                    ngraph::op::v1::Greater::get_type_info_static(),
                    ngraph::op::v1::GreaterEqual::get_type_info_static(),
                    ngraph::op::v1::Less::get_type_info_static(),
                    ngraph::op::v1::LessEqual::get_type_info_static(),
                    ngraph::op::v1::LogicalAnd::get_type_info_static(),
                    ngraph::op::v1::LogicalOr::get_type_info_static(),
                    ngraph::op::v1::LogicalXor::get_type_info_static(),
                    ngraph::op::v1::LogicalNot::get_type_info_static(),
                    ngraph::op::v0::Relu::get_type_info_static(),
                    ngraph::op::v0::Gelu::get_type_info_static(),
                    ngraph::op::v7::Gelu::get_type_info_static(),
                    ngraph::op::v0::Elu::get_type_info_static(),
                    ngraph::op::v0::Tanh::get_type_info_static(),
                    ngraph::op::v0::Sigmoid::get_type_info_static(),
                    ngraph::op::v0::Abs::get_type_info_static(),
                    ngraph::op::v0::Sqrt::get_type_info_static(),
                    ngraph::op::v0::Clamp::get_type_info_static(),
                    ngraph::op::v0::Exp::get_type_info_static(),
                    ngraph::op::v4::HSwish::get_type_info_static(),
                    ngraph::op::v4::Mish::get_type_info_static(),
                    ngraph::op::v5::HSigmoid::get_type_info_static(),
                    ngraph::op::v5::Round::get_type_info_static(),
                    ngraph::op::v0::PRelu::get_type_info_static(),
                    ngraph::op::v0::Erf::get_type_info_static(),
                    ngraph::op::v4::SoftPlus::get_type_info_static()))
            return false;

        return true;
    };

    manager.register_pass<ngraph::pass::FakeQuantizeDecomposition>();
    // Always decompose FQ if not fused and if not binarization
    manager.get_pass_config()->set_callback<ngraph::pass::FakeQuantizeDecomposition>(
        [isBinarizationFQ, FQMostLikelyToBeFused](const std::shared_ptr<const ngraph::Node> &node) -> bool {
            return isBinarizationFQ(node) || FQMostLikelyToBeFused(node);
        });

    manager.run_passes(nGraphFunc);
}

}  // namespace MKLDNNPlugin
