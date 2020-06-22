// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp"

#include "transformations/convert_gelu.hpp"
#include "transformations/convert_batch_to_space.hpp"
#include "transformations/convert_space_to_batch.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

bool ngraph::pass::ConvertOpSet2ToOpSet1::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;
    std::vector<std::shared_ptr<ngraph::pass::PassBase> > transforms;
    std::shared_ptr<ngraph::pass::GraphRewrite> anchor;

#include <transformations/utils/define_ngraph_pass.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1_tbl.hpp>
#include <transformations/utils/undefine_ngraph_pass.hpp>

    for (auto & t : transforms) {
        if (auto t_param = std::dynamic_pointer_cast<PassParam>(t)) {
            t_param->setCallback(transformation_callback);
        }
    }
    manager.run_passes(f);
    return true;
}