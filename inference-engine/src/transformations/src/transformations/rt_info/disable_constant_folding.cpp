// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_constant_folding.hpp"

template class ov::VariantImpl<ov::DisableConstantFolding>;

constexpr ov::VariantTypeInfo ov::VariantWrapper<ov::DisableConstantFolding>::type_info;

void ov::disable_constant_folding(const std::shared_ptr<Node>& node) {
    auto & rt_info = node->get_rt_info();
    rt_info[VariantWrapper<DisableConstantFolding>::type_info.name] = make_variant<DisableConstantFolding>({});
}
