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

//namespace ngraph {
//namespace pass {

class TRANSFORMATIONS_API MOCTransformations;

//}  // namespace pass
//}  // namespace ngraph

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
