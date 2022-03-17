// Copyright (C) 2018-2022 Intel Corporation
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
#include <ngraph/attribute_visitor.hpp>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/rtti.hpp>
#include <set>
#include <string>

#include "openvino/core/runtime_attribute.hpp"

namespace ngraph {

/**
 * @ingroup ie_runtime_attr_api
 * @brief FusedName class represents runtime info attribute that stores
 * all operation names that was fully or partially fused into node
 */
class NGRAPH_API FusedNames : public ov::RuntimeAttribute {
    std::set<std::string> fused_names;

public:
    OPENVINO_RTTI("fused_names", "0");

    /**
     * A default constructor
     */
    FusedNames() = default;

    /**
     * @brief      Constructs a new object consisting of a single name     *
     * @param[in]  name  The name
     */
    explicit FusedNames(const std::string& name) {
        fused_names.insert(name);
    }

    /**
     * @brief Unites current set of already fused names with another FusedNames object
     * @param[in] names Another object to fuse with
     */
    void fuseWith(const FusedNames& names);

    /**
     * @brief return string with operation names separated by coma in alphabetical order
     */
    std::string getNames() const;

    /**
     * @brief return vector of fused names sorted in alphabetical order
     * @return vector if strings
     */
    std::vector<std::string> getVectorNames() const;

    ov::Any merge(const ngraph::NodeVector& nodes) const override;

    ov::Any init(const std::shared_ptr<ngraph::Node>& node) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::string to_string() const override;
};

/**
 * @ingroup ie_runtime_attr_api
 * @brief getFusedNames return string with operation names separated by coma in alphabetical order
 * @param[in] node The node will be used to get FusedNames attribute
 */
NGRAPH_API std::string getFusedNames(const std::shared_ptr<ngraph::Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief getFusedNamesVector return vector of fused names sorted in alphabetical order
 * @param[in] node The node will be used to get FusedNames attribute
 * @return vector of strings
 */
NGRAPH_API std::vector<std::string> getFusedNamesVector(const std::shared_ptr<ngraph::Node>& node);

}  // namespace ngraph
