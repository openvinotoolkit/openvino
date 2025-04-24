// Copyright (C) 2018-2025 Intel Corporation
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
#include <set>
#include <string>

#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

/**
 * @ingroup ov_runtime_attr_api
 * @brief FusedName class represents runtime info attribute that stores
 * all operation names that was fully or partially fused into node
 */
class TRANSFORMATIONS_API FusedNames : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("fused_names", "0", RuntimeAttribute);

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

    ov::Any merge(const ov::NodeVector& nodes) const override;

    ov::Any init(const std::shared_ptr<ov::Node>& node) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::string to_string() const override;

private:
    std::set<std::string> fused_names;
};

/**
 * @ingroup ov_runtime_attr_api
 * @brief getFusedNames return string with operation names separated by coma in alphabetical order
 * @param[in] node The node will be used to get FusedNames attribute
 */
TRANSFORMATIONS_API std::string getFusedNames(const std::shared_ptr<ov::Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief getFusedNamesVector return vector of fused names sorted in alphabetical order
 * @param[in] node The node will be used to get FusedNames attribute
 * @return vector of strings
 */
TRANSFORMATIONS_API std::vector<std::string> getFusedNamesVector(const std::shared_ptr<ov::Node>& node);

}  // namespace ov
