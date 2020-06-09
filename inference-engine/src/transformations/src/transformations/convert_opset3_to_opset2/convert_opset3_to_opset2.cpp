// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp"

#include "transformations/convert_opset3_to_opset2/convert_broadcast3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_nms3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_shapeof3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_shuffle_channels3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_topk3.hpp"
#include "transformations/convert_extract_image_patches_to_reorg_yolo.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

bool ngraph::pass::ConvertOpSet3ToOpSet2::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager OpSet3ToOpSet2;
    std::vector<std::shared_ptr<ngraph::pass::PassBase> > transforms;

#define NGRAPH_PASS(NAME, NAMESPACE) transforms.push_back(OpSet3ToOpSet2.register_pass<NAMESPACE::NAME>());
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2_tbl.hpp>
#undef NGRAPH_PASS

    for (auto & t : transforms) {
        if (auto t_param = std::dynamic_pointer_cast<PassParam>(t)) {
            t_param->setCallback(transformation_callback);
        }
    }
    OpSet3ToOpSet2.run_passes(f);
    return true;
}
