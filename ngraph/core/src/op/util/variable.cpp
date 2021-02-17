// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/util/variable.hpp>

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<std::shared_ptr<Variable>>::type_info;
}
