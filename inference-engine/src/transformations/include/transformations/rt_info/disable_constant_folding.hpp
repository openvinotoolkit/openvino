// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>


namespace ov {

/**
 * @ingroup ie_runtime_attr_api
 * @brief DisableConstantFolding disable ConstantFolding for given operation
 */
class TRANSFORMATIONS_API DisableConstantFolding {
public:
    DisableConstantFolding() = default;
};

TRANSFORMATIONS_API void disable_constant_folding(const std::shared_ptr<Node>& node);

extern template class TRANSFORMATIONS_API VariantImpl<DisableConstantFolding>;

template<>
class TRANSFORMATIONS_API VariantWrapper<DisableConstantFolding> : public VariantImpl<DisableConstantFolding> {
public:
    OPENVINO_RTTI("disabled_constant_folding", "0");

    VariantWrapper() = default;

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override { return false; }
};

}  // namespace ov
