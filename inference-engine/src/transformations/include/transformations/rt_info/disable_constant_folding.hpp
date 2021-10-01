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

TRANSFORMATIONS_API void disable_constant_folding(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_constant_folding(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool constant_folding_is_disabled(const std::shared_ptr<Node>& node);

class TRANSFORMATIONS_API DisableConstantFolding : public VariantImpl<bool> {
public:
    OPENVINO_RTTI("disabled_constant_folding", "0");

    DisableConstantFolding() = default;

    DisableConstantFolding(const value_type &value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override { return false; }
};

}  // namespace ov
