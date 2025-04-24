// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/attributes.hpp"

ov::pass::Attributes::Attributes() {
    register_factory<ov::FusedNames>();
    register_factory<PrimitivesPriority>();
    register_factory<DisableConstantFolding>();
    register_factory<DisableFP16Compression>();
    register_factory<NmsSelectedIndices>();
    register_factory<OldApiMapOrder>();
    register_factory<OldApiMapElementType>();
    register_factory<LayoutAttribute>();
    register_factory<Decompression>();
    register_factory<ov::preprocess::TensorInfoMemoryType>();
    register_factory<StridesPropagation>();
    register_factory<PreprocessingAttribute>();
}

ov::Any ov::pass::Attributes::create_by_type_info(const ov::DiscreteTypeInfo& type_info) {
    auto it_type = m_factory_registry.find(type_info);
    if (it_type != m_factory_registry.end()) {
        return it_type->second();
    } else {
        return {};
    }
}
