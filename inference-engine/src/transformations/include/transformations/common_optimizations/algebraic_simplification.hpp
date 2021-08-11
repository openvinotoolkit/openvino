// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/pass.hpp>

#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AlgebraicSimplification;

}  // namespace pass
}  // namespace ov

class ov::pass::AlgebraicSimplification : public GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    AlgebraicSimplification() = default;
};
