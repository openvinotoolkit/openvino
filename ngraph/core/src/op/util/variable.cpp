// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/util/variable.hpp>

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<std::shared_ptr<ngraph::Variable>>::type_info;
