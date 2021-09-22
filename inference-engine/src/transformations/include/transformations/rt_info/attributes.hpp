// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include <transformations_visibility.hpp>

#include <openvino/core/variant.hpp>
#include <ngraph/node.hpp>
#include <ngraph/factory.hpp>

#include <transformations/rt_info/disable_constant_folding.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/rt_info/nms_selected_indices.hpp>
#include <transformations/rt_info/primitives_priority_attribute.hpp>
#include <transformations/rt_info/strides_property.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API Attributes {
public:
    Attributes() {
        register_factory<VariantWrapper<ngraph::FusedNames>>();
    }

    Variant * create(const std::string & name, const uint64_t version) {
        ov::DiscreteTypeInfo info(name.c_str(), version);
        return m_factory_registry.create(info);
    }
private:
    template <class T>
    void register_factory() {
        m_factory_registry.register_factory<T>(ngraph::FactoryRegistry<T>::template get_default_factory<T>());
    }

    ngraph::FactoryRegistry<Variant> m_factory_registry;
};
} // namespace pass
} // namespace ov