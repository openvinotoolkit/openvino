// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <ngraph/factory.hpp>
#include <ngraph/node.hpp>
#include <openvino/core/variant.hpp>
#include <set>
#include <transformations/rt_info/disable_constant_folding.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/rt_info/nms_selected_indices.hpp>
#include <transformations/rt_info/old_api_map_attribute.hpp>
#include <transformations/rt_info/primitives_priority_attribute.hpp>
#include <transformations/rt_info/strides_property.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API Attributes {
public:
    Attributes();
    ~Attributes();

    Variant* create_by_type_info(const ov::DiscreteTypeInfo& type_info);

private:
    template <class T>
    void register_factory() {
        m_factory_registry.register_factory<T>(ngraph::FactoryRegistry<T>::template get_default_factory<T>());
    }

    ngraph::FactoryRegistry<Variant> m_factory_registry;
};
}  // namespace pass
}  // namespace ov
