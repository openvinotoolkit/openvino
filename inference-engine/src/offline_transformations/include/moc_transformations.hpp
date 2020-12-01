// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <cpp/ie_cnn_network.h>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/remove_filtering_boxes_by_size.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/common_optimizations/algebraic_simplification.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/mish_fusion.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>

#include <disable_constant_folding.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>

#include <ngraph/opsets/opset4.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <transformations/op_conversions/convert_convolutions.hpp>
#include <transformations/common_optimizations/conv_bias_fusion.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>

class TRANSFORMATIONS_API MOCTransformations;
class TRANSFORMATIONS_API MOCTransformationsCPU;

class MOCTransformations: public ngraph::pass::FunctionPass {
    bool m_cf;

public:
    NGRAPH_RTTI_DECLARATION;
    explicit MOCTransformations(bool cf) : m_cf(cf) {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        std::cout << "Hello MOC! " << (m_cf ? "ShapeOf disabled" : "ShapeOf enabled") << std::endl;

        ngraph::pass::Manager manager(get_pass_config());

        auto disable_cf = manager.register_pass<ngraph::pass::GraphRewrite>();
        if (!m_cf) {
            disable_cf->add_matcher<ngraph::pass::DisableShapeOfConstantFolding>();
        } else {
            disable_cf->add_matcher<ngraph::pass::DisablePriorBoxConstantFolding>();
            disable_cf->add_matcher<ngraph::pass::DisablePriorBoxClusteredConstantFolding>();
        }
        disable_cf->set_name("ngraph::pass::DisableConstantFolding");

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
        manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // depends on CF
        manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
        manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ConvertScatterElementsToScatter>(); // partially depends on CF
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.register_pass<ngraph::pass::MishFusion>();
        manager.register_pass<ngraph::pass::SoftPlusFusion>();
        manager.register_pass<ngraph::pass::SoftPlusToMishFusion>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.register_pass<ngraph::pass::HSigmoidFusion>();
        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
        manager.register_pass<ngraph::pass::ConstantFolding>();

        manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

        auto conv_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
        conv_fusions->add_matcher<ngraph::pass::ConvolutionMultiplyFusion>();
        conv_fusions->add_matcher<ngraph::pass::GroupConvolutionMultiplyFusion>();
        conv_fusions->add_matcher<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
        conv_fusions->add_matcher<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
        conv_fusions->set_name("ngraph::pass::ConvFusions");

        manager.register_pass<ngraph::pass::ConstantFolding>();

        auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
        fq_fusions->add_matcher<ngraph::pass::FakeQuantizeMulFusion>();
        fq_fusions->add_matcher<ngraph::pass::FakeQuantizeReshapeFusion>();
        fq_fusions->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
        fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");

        manager.register_pass<ngraph::pass::EnableShapeOfConstantFolding>();
        manager.run_passes(f);

        return true;
    }
};

class MOCTransformationsCPU: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        std::cout << "Hello MOC for CPU! " << std::endl;

        ngraph::pass::Manager manager(get_pass_config());

        auto disable_cf = manager.register_pass<ngraph::pass::GraphRewrite>();
        disable_cf->add_matcher<ngraph::pass::DisablePriorBoxConstantFolding>();
        disable_cf->add_matcher<ngraph::pass::DisablePriorBoxClusteredConstantFolding>();
        disable_cf->set_name("ngraph::pass::DisableConstantFolding");

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
//        manager.register_pass<ngraph::pass::ConvertPriorBox>();
        manager.register_pass<ngraph::pass::ConvertNMS5ToLegacyMatcher>();
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        manager.register_pass<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
        manager.register_pass<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
        manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
        manager.register_pass<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
        manager.register_pass<ngraph::pass::GRUCellDecomposition>();
        manager.register_pass<ngraph::pass::RNNCellDecomposition>();

        std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> convert_precision_list{
                {ngraph::element::i64,     ngraph::element::i32},
                {ngraph::element::u64,     ngraph::element::i32},
                {ngraph::element::u16,     ngraph::element::i32},
                {ngraph::element::u32,     ngraph::element::i32},
                {ngraph::element::f16,     ngraph::element::f32},
                {ngraph::element::boolean, ngraph::element::u8},
        };

        for (auto &precision : convert_precision_list) {
            manager.register_pass<ngraph::pass::ConvertPrecision>(precision.first, precision.second);
        }

        auto pass_config = manager.get_pass_config();

        using const_node_ptr = const std::shared_ptr<const ngraph::Node>;

        // SpaceToDepth/ DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
        pass_config->set_callback<ngraph::pass::ConvertSpaceToDepth,
                ngraph::pass::ConvertDepthToSpace>(
                [](const_node_ptr &node) -> bool {
                    return node->input_value(0).get_shape().size() <= 5lu &&
                           node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
                });

        // Disable FC reshaping for 3D case
        pass_config->set_callback<ngraph::pass::ReshapeFullyConnected>(
                [](const_node_ptr &node) -> bool {
                    return node->input_value(0).get_shape().size() == 3ul;
                });

        pass_config->set_callback<ngraph::pass::ConvertBatchToSpace,
                                  ngraph::pass::ConvertSpaceToBatch>(
                [](const_node_ptr &node) -> bool {
                    const auto & rank = node->input(0).get_partial_shape().rank().get_length();
                    return rank == 4lu || rank == 5lu;
                });

        auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
            if (const auto &rnn_cell = std::dynamic_pointer_cast<const ngraph::opset4::RNNCell>(node)) {
                return rnn_cell->get_clip() == 0.0f;
            } else if (const auto &gru_cell = std::dynamic_pointer_cast<const ngraph::opset4::GRUCell>(
                    node)) {
                return gru_cell->get_clip() == 0.0f
                       && gru_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh"};
            } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ngraph::opset4::LSTMCell>(
                    node)) {
                return lstm_cell->get_clip() == 0.0f &&
                       lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ngraph::opset1::LSTMCell>(
                    node)) {
                return lstm_cell_v1->get_clip() == 0.0f &&
                       lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            }
            return false;
        };

        pass_config->set_callback<ngraph::pass::RNNCellDecomposition, ngraph::pass::GRUCellDecomposition,
                ngraph::pass::LSTMCellDecomposition>(
                [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                    return isCellPrimitiveSupported(node);
                });

        pass_config->set_callback<ngraph::pass::ConvertTensorIteratorToRNNSequence,
                                  ngraph::pass::ConvertTensorIteratorToLSTMSequence,
                                  ngraph::pass::ConvertTensorIteratorToGRUSequence>(
                [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                    if (const auto& ti_op = std::dynamic_pointer_cast<const ngraph::op::TensorIterator>(node)) {
                        size_t count_rnn = 0;
                        for (const auto &op : ti_op->get_body()->get_ops())
                            count_rnn += isCellPrimitiveSupported(op);
                        return count_rnn != 1;
                    }
                    return true;
                });

        // List of enabled/disabled transformations
        pass_config->disable<ngraph::pass::ConvertGELU>();
        pass_config->disable<ngraph::pass::HSwishDecomposition>();
        pass_config->disable<ngraph::pass::ReduceL1Decomposition>();
        pass_config->disable<ngraph::pass::ReduceL2Decomposition>();
        pass_config->disable<ngraph::pass::SoftPlusDecomposition>();
        pass_config->disable<ngraph::pass::HSigmoidDecomposition>();
        pass_config->disable<ngraph::pass::ConvertMod>();
        pass_config->disable<ngraph::pass::LogSoftmaxDecomposition>();

        pass_config->enable<ngraph::pass::ConvertPadToGroupConvolution>();

        // Convolution/Deconvolution/FullyConnected fusions
        auto convert_convolutions = manager.register_pass<ngraph::pass::GraphRewrite>();
        convert_convolutions->add_matcher<ngraph::pass::ConvertMatMulToFC>();
        convert_convolutions->add_matcher<ngraph::pass::ConvertConvolution>();
        convert_convolutions->add_matcher<ngraph::pass::ConvertGroupConvolution>();
        convert_convolutions->add_matcher<ngraph::pass::ConvertDeconvolution>();
        convert_convolutions->add_matcher<ngraph::pass::ConvertGroupDeconvolution>();
        convert_convolutions->set_name("ngraph::pass::ConvertConvolutions");

        // Convolution/Deconvolution/FullyConnected fusions
        auto fusion = manager.register_pass<ngraph::pass::GraphRewrite>();
        fusion->add_matcher<ngraph::pass::ConvAddFusion>();
        fusion->add_matcher<ngraph::pass::DeconvAddFusion>();
        fusion->add_matcher<ngraph::pass::FullyConnectedBiasFusion>();
        fusion->set_name("ngraph::pass::BiasFusions");

        manager.register_pass<ngraph::pass::ConstantFolding>();

//        manager.register_pass<ngraph::pass::Serialize>("/home/gkazanta/openvino/model-optimizer/./resnest_cpu.xml",
//                                                   "/home/gkazanta/openvino/model-optimizer/./resnest_cpu.bin");
        manager.register_pass<ngraph::pass::EnableShapeOfConstantFolding>();
        manager.run_passes(f);
        return true;
    }
};
