// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/convert_fp32_to_fp16.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "transformations/convert_precision.hpp"

using namespace std;

bool ov::pass::ConvertFP32ToFP16::run_on_model(const std::shared_ptr<ov::Model>& f) {
    ov::pass::Manager m(get_pass_config());
    m.register_pass<ngraph::pass::ConvertPrecision>(precisions_array{{ngraph::element::f32, ngraph::element::f16}});
    m.run_passes(f);
    return false;
}
