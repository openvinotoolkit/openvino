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
 * @brief FusedName class represents runtime info attribute that stores
 * all operation names that was fully or partially fused into node
 */
class TRANSFORMATIONS_API FusedNames {
private:
    std::set<std::string> fused_names;

public:
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
     * @param names[in] Another object to fuse with
     */
    void fuseWith(const FusedNames &names);

    // return string with operation names separated by coma in alphabetical order
    std::string getNames() const;

    // return vector of fused names sorted in alphabetical order
    std::vector<std::string> getVectorNames() const;
};

template<>
class TRANSFORMATIONS_API VariantWrapper<FusedNames> : public VariantImpl<FusedNames> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::FusedNames", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;
};

/**
 * @ingroup ie_runtime_attr_api
 * @brief getFusedNames return string with operation names separated by coma in alphabetical order
 */
TRANSFORMATIONS_API std::string getFusedNames(const std::shared_ptr<ngraph::Node> & node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief getFusedNamesVector return vector of fused names sorted in alphabetical order
 */
TRANSFORMATIONS_API std::vector<std::string> getFusedNamesVector(const std::shared_ptr<ngraph::Node> & node);

}  // namespace ngraph
