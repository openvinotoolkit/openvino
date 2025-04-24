// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "openvino/core/any.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/nms_selected_indices.hpp"
#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"
#include "transformations/rt_info/old_api_map_order_attribute.hpp"
#include "transformations/rt_info/preprocessing_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "transformations/rt_info/strides_property.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API Attributes {
public:
    Attributes();

    Any create_by_type_info(const ov::DiscreteTypeInfo& type_info_name);

private:
    template <class T>
    void register_factory() {
        m_factory_registry.emplace(T::get_type_info_static(), []() -> Any {
            return T{};
        });
    }

    std::unordered_map<ov::DiscreteTypeInfo, std::function<Any()>> m_factory_registry;
};
}  // namespace pass
}  // namespace ov
