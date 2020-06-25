// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/depth_to_space_fusion.hpp"
#include "transformations/optimize_strided_slice.hpp"
#include "transformations/convert_scatter_elements_to_scatter.hpp"
#include "transformations/remove_filtering_boxes_by_size.hpp"
#include "transformations/convert_nms_4_to_nms_dynamic.hpp"
#include "transformations/convert_nms_4_to_legacy.hpp"
#include "transformations/init_node_info.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/nop_elimination.hpp>
#include <ngraph/pass/algebraic_simplification.hpp>
#include <ngraph/pass/constant_folding.hpp>


bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager CommonOptimizations;
    std::vector<std::shared_ptr<ngraph::pass::PassBase> > transforms;

#define NGRAPH_PASS(NAME, NAMESPACE) transforms.push_back(CommonOptimizations.register_pass<NAMESPACE::NAME>());
#include <transformations/common_optimizations/common_optimizations_tbl.hpp>
#undef NGRAPH_PASS

    for (auto & t : transforms) {
        if (auto t_param = std::dynamic_pointer_cast<PassParam>(t)) {
            t_param->setCallback(transformation_callback);
        }
    }
    CommonOptimizations.run_passes(f);
    return true;
}
