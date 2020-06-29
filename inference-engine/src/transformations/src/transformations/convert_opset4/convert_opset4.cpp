// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset4/convert_opset4.hpp"

#include "transformations/convert_opset4/convert_nms_4_to_legacy.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

bool ngraph::pass::ConvertOpSet4::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager OpSet4Manager;
    std::vector<std::shared_ptr<ngraph::pass::PassBase> > transforms;

#define NGRAPH_PASS(NAME, NAMESPACE) transforms.push_back(OpSet4Manager.register_pass<NAMESPACE::NAME>());
#include <transformations/convert_opset4/convert_opset4_tbl.hpp>
#undef NGRAPH_PASS

    for (auto & t : transforms) {
        if (auto t_param = std::dynamic_pointer_cast<PassParam>(t)) {
            t_param->setCallback(transformation_callback);
        }
    }
    OpSet4Manager.run_passes(f);
    return true;
}
