// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"

namespace ngraph {
namespace pass {

class ConvertMulAddToScaleShiftOrPower;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulAddToScaleShiftOrPower: public ngraph::pass::GraphRewrite {
public:
    ConvertMulAddToScaleShiftOrPower() : GraphRewrite() {
        convert_mul_add_to_scaleshift_or_power();
    }

private:
    void convert_mul_add_to_scaleshift_or_power();
};

enum class CONVERSION_RESULT {
    SCALE_SHIFT,
    POWER,
    NONE
};

CONVERSION_RESULT check_constant(const std::shared_ptr<ngraph::op::Constant> & constant,
                                 const std::vector<int64_t> & output_shape);

CONVERSION_RESULT check_dyn_broadcast(const std::shared_ptr<ngraph::op::DynBroadcast> & broadcast);

std::shared_ptr<ngraph::Node> normalize_constant(const std::shared_ptr<ngraph::op::Constant> & constant,
                                                 const ngraph::Shape & shape);
