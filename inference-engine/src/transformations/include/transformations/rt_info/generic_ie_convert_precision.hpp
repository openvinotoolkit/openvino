// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines fused names attribute
 * @file fused_names_attribute.hpp
 */

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
 * @brief GenericIEConvertPrecision is a legacy attribute type needed only for
 * connection between GenericIE and ngraph::ConvertPrecision transformation.
 */
class TRANSFORMATIONS_API GenericIEConvertPrecision {
public:
    using callback = std::function<void(const element::Type&)>;

    explicit GenericIEConvertPrecision(const callback & m_callback)
        : convert_precision_callback(m_callback) {}

    void convert_precision(const element::Type & type) {
        convert_precision_callback(type);
    }

private:
    callback convert_precision_callback;
};

extern template class TRANSFORMATIONS_API VariantImpl<GenericIEConvertPrecision>;

template<>
class TRANSFORMATIONS_API VariantWrapper<GenericIEConvertPrecision> : public VariantImpl<GenericIEConvertPrecision> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::GenericIEConvertPrecision", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}
};

}  // namespace ngraph
