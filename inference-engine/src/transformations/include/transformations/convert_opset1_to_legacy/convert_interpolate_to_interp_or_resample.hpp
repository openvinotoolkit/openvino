// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertInterpolateToInterpOrResample);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertInterpolateToInterpOrResample: public ngraph::pass::GraphRewrite {
public:
    ConvertInterpolateToInterpOrResample() : GraphRewrite() {
        convert_interpolate_to_interp_or_resample();
    }

private:
    void convert_interpolate_to_interp_or_resample();
};
