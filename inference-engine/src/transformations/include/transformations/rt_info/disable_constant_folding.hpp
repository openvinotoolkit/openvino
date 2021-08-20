// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>


namespace ngraph {

/**
 * @ingroup ie_runtime_attr_api
 * @brief DisableConstantFolding disable ConstantFolding for given operation
 */
class TRANSFORMATIONS_API DisableConstantFolding {
public:
    DisableConstantFolding() = default;
};

TRANSFORMATIONS_API void disable_constant_folding(const std::shared_ptr<Node>& node);
}  // namespace ngraph

namespace ov {
extern template class TRANSFORMATIONS_API VariantImpl<ngraph::DisableConstantFolding>;

template<>
class TRANSFORMATIONS_API VariantWrapper<ngraph::DisableConstantFolding> : public VariantImpl<ngraph::DisableConstantFolding> {
public:
    static constexpr VariantTypeInfo type_info{"DISABLED_CONSTANT_FOLDING", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override { return false; }
};

}  // namespace ov
