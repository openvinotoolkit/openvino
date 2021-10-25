// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/attributes.hpp"

ov::pass::Attributes::Attributes() {
    register_factory<VariantWrapper<ngraph::FusedNames>>();
    register_factory<PrimitivesPriority>();
    register_factory<DisableConstantFolding>();
    register_factory<NmsSelectedIndices>();
    register_factory<StridesPropagation>();
    register_factory<OldApiMap>();
}

ov::Variant* ov::pass::Attributes::create_by_type_info(const ov::DiscreteTypeInfo& type_info) {
    return m_factory_registry.create(type_info);
}

ov::pass::Attributes::~Attributes() = default;
