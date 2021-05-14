// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/convert_fp32_to_fp16.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "transformations/convert_precision.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertFP32ToFP16, "ConvertFP32ToFP16", 0);

bool ngraph::pass::ConvertFP32ToFP16::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    ngraph::pass::Manager m(get_pass_config());
    m.register_pass<ngraph::pass::ConvertPrecision>(
        precisions_array{{ngraph::element::f32, ngraph::element::f16}});
    m.run_passes(f);
    return false;
}
