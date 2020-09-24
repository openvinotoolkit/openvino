// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/common_optimizations/algebraic_simplification.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/depth_to_space_fusion.hpp"
#include "transformations/optimize_strided_slice.hpp"
#include "transformations/convert_scatter_elements_to_scatter.hpp"
#include "transformations/convert_pad_to_group_conv.hpp"
#include "transformations/remove_filtering_boxes_by_size.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/itt.hpp"
#include "transformations/mish_fusion.hpp"
#include "transformations/softplus_fusion.hpp"
#include "transformations/softplus_to_mish_fusion.hpp"
#include "transformations/swish_fusion.hpp"
#include "transformations/hswish_fusion.hpp"
#include "transformations/normalize_l2_fusion.hpp"
#include "transformations/convert_quantize_dequantize.hpp"
#include "transformations/bidirectional_sequences_decomposition.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::CommonOptimizations, "CommonOptimizations", 0);

bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::CommonOptimizations");

    ngraph::pass::Manager manager;

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // depends on CF
    manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertScatterElementsToScatter>(); // partially depends on CF
    manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    manager.register_pass<ngraph::pass::MishFusion>();
    manager.register_pass<ngraph::pass::SoftPlusFusion>();
    manager.register_pass<ngraph::pass::SoftPlusToMishFusion>();
    manager.register_pass<ngraph::pass::SwishFusion>();
    manager.register_pass<ngraph::pass::HSwishFusion>();
    manager.register_pass<ngraph::pass::ConvertPadToGroupConvolution>();
    manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    manager.register_pass<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
    manager.register_pass<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
    manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
