// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <assert.h>
#include <vector>
#include <limits>

#include <precision_utils.h>
#include <ngraph/op/util/op_annotations.hpp>
#include <ngraph/op/constant.hpp>
#include <details/ie_exception.hpp>

namespace ngraph {
namespace op {
namespace util {

template <class T>
bool normalize_single_value(std::vector<T> vec, float & value) {
    for (const auto & val : vec) {
        if (val != *vec.begin()) return false;
    }

    float ref_val = *vec.begin();

    if (ref_val < std::numeric_limits<float>::lowest() || ref_val > std::numeric_limits<float>::max()) {
        return false;
    }

    value = ref_val;
    return true;
}

bool get_single_value(const std::shared_ptr<op::Constant> & const_node, float & value);

std::shared_ptr<ngraph::Node> normalize_constant(const std::shared_ptr<op::Constant> & constant,
                                                 const Shape & shape);

std::shared_ptr<ngraph::Node> broadcastTo(const Output<Node>& input, const Shape& shape);

std::shared_ptr<ngraph::Node> reshapeTo(const Output<Node> & input, const Shape& shape);

bool constantIsEqualTo(const std::shared_ptr<ngraph::op::Constant>& const_node, float value, float eps = 1e-5);

}  // namespace util
}  // namespace op
}  // namespace ngraph
