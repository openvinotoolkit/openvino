// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"

#include "transformations/op_conversions/convert_broadcast3.hpp"
#include "transformations/op_conversions/convert_shapeof3.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_topk3.hpp"
#include "transformations/op_conversions/softplus_decomposition.hpp"
#include "transformations/itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertOpSet3ToOpSet2, "ConvertOpSet3ToOpSet2", 0);

bool ngraph::pass::ConvertOpSet3ToOpSet2::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager(get_pass_config());

    manager.register_pass<ngraph::pass::ConvertBroadcast3>();
    manager.register_pass<ngraph::pass::ConvertShapeOf3>();
    manager.register_pass<ngraph::pass::ConvertShuffleChannels3>();
    manager.register_pass<ngraph::pass::ConvertTopK3>();
    manager.register_pass<ngraph::pass::SoftPlusDecomposition>();

    manager.run_passes(f);
    return true;
}
