// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "moc_transformations.hpp"
#include "disable_shapeof_constant_folding.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/gelu_fusion.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/common_optimizations/remove_filtering_boxes_by_size.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/common_optimizations/clamp_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/common_optimizations/binarize_weights.hpp>
#include <transformations/common_optimizations/conv_to_binary_conv.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <transformations/common_optimizations/leaky_relu_fusion.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MOCTransformations, "MOCTransformations", 0);

bool ngraph::pass::MOCTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // To avoid issues with dynamism we make nGraph Function dynamic and after we apply all
    // transformations we restore original shapes to the nGraph Function back
    std::unordered_map<ngraph::op::Parameter*, PartialShape> input_shapes;
    for (auto && param : f->get_parameters()) {
        input_shapes[param.get()] = param->get_partial_shape();
        param->set_partial_shape(PartialShape::dynamic(param->get_partial_shape().rank()));
    }
    f->validate_nodes_and_infer_types();

    ngraph::pass::Manager manager(get_pass_config());

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
            element::TypeVector{ ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4 });
    manager.register_pass<ngraph::pass::DisableShapeOfConstantFolding>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>();
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::SimplifyShapeOfSubGraph>();

    auto transpose_sinking = manager.register_pass<ngraph::pass::GraphRewrite>();
    transpose_sinking->add_matcher<ngraph::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    transpose_sinking->add_matcher<ngraph::pass::SplitSqueezeConcatFusion>();

    auto eliminations = manager.register_pass<ngraph::pass::GraphRewrite>();
    eliminations->add_matcher<ngraph::pass::EliminateUnsqueezeGather>();
    eliminations->add_matcher<ngraph::pass::NopElimination>(false /* do not use shape for elimination */);
    eliminations->set_name("ngraph::pass::CommonEliminations");

    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    common_fusions->add_matcher<ngraph::pass::ConvertScatterElementsToScatter>();
    common_fusions->add_matcher<ngraph::pass::BroadcastElementwiseFusion>();
    common_fusions->add_matcher<ngraph::pass::SoftPlusFusion>();
    common_fusions->add_matcher<ngraph::pass::SoftPlusToMishFusion>();
    common_fusions->add_matcher<ngraph::pass::SwishFusion>();
    common_fusions->add_matcher<ngraph::pass::HSwishFusion>();
    common_fusions->add_matcher<ngraph::pass::HSigmoidFusion>();
    common_fusions->add_matcher<ngraph::pass::NormalizeL2Fusion>();
    common_fusions->add_matcher<ngraph::pass::ClampFusion>();
    common_fusions->add_matcher<ngraph::pass::PadFusion>();
    common_fusions->add_matcher<ngraph::pass::MVNFusion>();
    common_fusions->add_matcher<ngraph::pass::DilatedConvolutionConverter>();
    common_fusions->add_matcher<ngraph::pass::GeluFusion>();
    common_fusions->add_matcher<ngraph::pass::LeakyReluFusion>();
    common_fusions->set_name("ngraph::pass::CommonFusions");

    manager.register_pass<ngraph::pass::BinarizeWeights>();
    manager.register_pass<ngraph::pass::ConvToBinaryConv>();

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();

    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

    auto conv_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    conv_fusions->add_matcher<ngraph::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->set_name("ngraph::pass::ConvFusions");

    manager.run_passes(f);

    // Restore original shapes to the nGraph Function
    for (auto && param : f->get_parameters()) {
        param->set_partial_shape(input_shapes.at(param.get()));
    }
    f->validate_nodes_and_infer_types();

    return false;
}