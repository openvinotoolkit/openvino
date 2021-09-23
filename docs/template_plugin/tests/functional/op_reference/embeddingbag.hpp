// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#define ConstantPtr     std::shared_ptr<ngraph::opset1::Constant>
#define MakeConstantPtr std::make_shared<ngraph::opset1::Constant>

#include "base_reference_test.hpp"

template <class T>
inline ConstantPtr GetConstantVec(const std::vector<T>& val, const ngraph::element::Type& element_type) {
    return MakeConstantPtr(element_type, std::vector<size_t>{val.size()}, val);
}

template <class T>
inline ConstantPtr GetConstantVal(const T& val, const ngraph::element::Type& element_type) {
    return MakeConstantPtr(element_type, std::vector<size_t>{}, val);
}
