// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "moc_transformations.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/algebraic_simplification.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/gelu_fusion.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>
#include <transformations/common_optimizations/remove_filtering_boxes_by_size.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/clamp_fusion.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/softmax_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/binarize_weights.hpp>
#include <transformations/common_optimizations/conv_to_binary_conv.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/serialize.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MOCTransformations, "MOCTransformations", 0);

bool ngraph::pass::MOCTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // To avoid issues with dynamism we make ngraph::Function dynamic and in the end we change it back
    std::unordered_map<std::string, PartialShape> input_shapes;
    for (auto && param : f->get_parameters()) {
        input_shapes[param->get_friendly_name()] = param->get_partial_shape();
        param->set_partial_shape(PartialShape::dynamic(param->get_partial_shape().rank()));
    }
    f->validate_nodes_and_infer_types();

    ngraph::pass::Manager manager(get_pass_config());

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    // manager.register_pass<ngraph::pass::ConstantFolding>();
    // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // + RemoveFilteringBoxesBySize

    // TODO: move to KMB
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>(); // + QuantizeDequantizeLinear
    // manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>(); // - ??? do we need this in offline?

    // manager.register_pass<ngraph::pass::ConstantFolding>();
    // depends on CF
    // manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // - (ConvertGroupedStridedSlice - also relies on shape)
    // manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>(); // - keep in offline

    // auto transpose_sinking = manager.register_pass<ngraph::pass::GraphRewrite>(); - TBD
    // transpose_sinking->add_matcher<ngraph::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    // transpose_sinking->add_matcher<ngraph::pass::SplitSqueezeConcatFusion>();

//    auto eliminations = manager.register_pass<ngraph::pass::GraphRewrite>();
//    eliminations->add_matcher<ngraph::pass::EliminateUnsqueezeGather>();
//    eliminations->add_matcher<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
//    eliminations->add_matcher<ngraph::pass::NopElimination>(); // may introduce fake dynamism
//    eliminations->set_name("ngraph::pass::CommonEliminations");

    // manager.register_pass<ngraph::pass::ConstantFolding>();

    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
//    common_fusions->add_matcher<ngraph::pass::ConvertScatterElementsToScatter>(); // -

    // requires static shape
    // common_fusions->add_matcher<ngraph::pass::DepthToSpaceFusion>(); // +


    common_fusions->add_matcher<ngraph::pass::SoftPlusFusion>(); // + SoftplusFusion
    common_fusions->add_matcher<ngraph::pass::SoftPlusToMishFusion>(); // + MishFusion
    common_fusions->add_matcher<ngraph::pass::SwishFusion>(); // + SwishWithSigmoidWithoutBeta,
//    common_fusions->add_matcher<ngraph::pass::ShuffleChannelsFusion>(false); // + ShuffleChannelFusion
    common_fusions->add_matcher<ngraph::pass::HSwishFusion>(); // + HSwishWithClamp
    common_fusions->add_matcher<ngraph::pass::HSigmoidFusion>(); // + HSigmoidWithClamp...
//    common_fusions->add_matcher<ngraph::pass::NormalizeL2Fusion>(); // -
//    common_fusions->add_matcher<ngraph::pass::ClampFusion>(); // -
    common_fusions->add_matcher<ngraph::pass::PadFusion>(); // - RemoveUselessPad ???
//    common_fusions->add_matcher<ngraph::pass::SoftmaxFusion>(); // -
//    common_fusions->add_matcher<ngraph::pass::MVNFusion>(); // -
//    common_fusions->add_matcher<ngraph::pass::SpaceToBatchFusion>(); // -
//    common_fusions->add_matcher<ngraph::pass::BatchToSpaceFusion>(); // - BatchToSpaceToUpsample  ???
    common_fusions->add_matcher<ngraph::pass::DilatedConvolutionConverter>(); // + DilatedConvolutionConverter
    common_fusions->add_matcher<ngraph::pass::GeluFusion>(); // + GeLUMergerErf
    common_fusions->set_name("ngraph::pass::CommonFusions");

    // TODO: replace ConvToBinaryConv and BinarizeWeightsM1P1
    // manager.register_pass<ngraph::pass::BinarizeWeights>();
    // manager.register_pass<ngraph::pass::ConvToBinaryConv>();

    manager.run_passes(f);

    for (auto && param : f->get_parameters()) {
        param->set_partial_shape(input_shapes.at(param->get_friendly_name()));
    }
    f->validate_nodes_and_infer_types();

    return false;
}