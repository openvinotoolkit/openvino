// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines fused names attribute
 * @file fused_names_attribute.hpp
 */

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/rtti.hpp>
#include <ngraph/attribute_visitor.hpp>
#include <transformations_visibility.hpp>


namespace ov {

/**
 * @ingroup ie_runtime_attr_api
 * @brief FusedName class represents runtime info attribute that stores
 * all operation names that was fully or partially fused into node
 */
class TRANSFORMATIONS_API FusedNames {
private:
    std::set<std::string> fused_names;

public:
    friend class VariantWrapper<FusedNames>;

    /**
     * A default constructor
     */
    FusedNames() = default;

    /**
     * @brief      Constructs a new object consisting of a single name     *
     * @param[in]  name  The name
     */
    explicit FusedNames(const std::string &name) {
        fused_names.insert(name);
    }

    /**
     * @brief Unites current set of already fused names with another FusedNames object
     * @param[in] names Another object to fuse with
     */
    void fuseWith(const FusedNames &names);

    /**
     * @brief return string with operation names separated by coma in alphabetical order
     */
    std::string getNames() const;

    /**
     * @brief return vector of fused names sorted in alphabetical order
     * @return vector if strings
     */
    std::vector<std::string> getVectorNames() const;
};

/**
 * @ingroup ie_runtime_attr_api
 * @brief getFusedNames return string with operation names separated by coma in alphabetical order
 * @param[in] node The node will be used to get FusedNames attribute
 */
TRANSFORMATIONS_API std::string getFusedNames(const std::shared_ptr<ngraph::Node> & node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief getFusedNamesVector return vector of fused names sorted in alphabetical order
 * @param[in] node The node will be used to get FusedNames attribute
 * @return vector of strings
 */
TRANSFORMATIONS_API std::vector<std::string> getFusedNamesVector(const std::shared_ptr<ngraph::Node> & node);

extern template class TRANSFORMATIONS_API VariantImpl<FusedNames>;

template<>
class TRANSFORMATIONS_API VariantWrapper<FusedNames> : public VariantImpl<FusedNames> {
public:
    OPENVINO_RTTI("fused_names", "0");

    VariantWrapper() = default;

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;

    bool visit_attributes(AttributeVisitor & visitor) override;
};

template <>
class OPENVINO_API AttributeAdapter<std::set<std::string>> : public DirectValueAccessor<std::set<std::string>> {
public:
    OPENVINO_RTTI("AttributeAdapter<set<string>>");
    AttributeAdapter(std::set<std::string>& value) : DirectValueAccessor<std::set<std::string>>(value) {}
};
}  // namespace ov
