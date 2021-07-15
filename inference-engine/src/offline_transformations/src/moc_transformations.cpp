// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "moc_transformations.hpp"

#include <ngraph/pass/manager.hpp>
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
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>();
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::SimplifyShapeOfSubGraph>();

    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    common_fusions->add_matcher<ngraph::pass::SoftPlusFusion>();
    common_fusions->add_matcher<ngraph::pass::SoftPlusToMishFusion>();
    common_fusions->add_matcher<ngraph::pass::SwishFusion>();
    common_fusions->add_matcher<ngraph::pass::HSwishFusion>();
    common_fusions->add_matcher<ngraph::pass::HSigmoidFusion>();
    common_fusions->add_matcher<ngraph::pass::PadFusion>();
    common_fusions->add_matcher<ngraph::pass::GeluFusion>();
    common_fusions->set_name("ngraph::pass::CommonFusions");

    manager.run_passes(f);

    // Restore original shapes to the nGraph Function
    for (auto && param : f->get_parameters()) {
        param->set_partial_shape(input_shapes.at(param.get()));
    }
    f->validate_nodes_and_infer_types();

    return false;
}