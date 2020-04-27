// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ie_api.h>


namespace ngraph {

/*
 * Description:
 *     FusedName class represents runtime info attribute that stores
 *     all operation names that was fully or partially fused into node
 */

class FusedNames {
private:
    std::set<std::string> fused_names;

public:
    FusedNames() = default;

    explicit FusedNames(const std::string &name) {
        fused_names.insert(name);
    }

    // This method unite current set of already fused names with another FusedNames object
    void fuseWith(const FusedNames &names);

    // return string with operation names separated by coma in alphabetical order
    std::string getNames() const;

    // returns vector of fused names sorted in alphabetical order
    std::vector<std::string> getVectorNames() const;
};

template<>
class VariantWrapper<FusedNames> : public VariantImpl<FusedNames> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::FusedNames", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;
};

INFERENCE_ENGINE_API_CPP(std::string) getFusedNames(const std::shared_ptr<ngraph::Node> & node);

INFERENCE_ENGINE_API_CPP(std::vector<std::string>) getFusedNamesVector(const std::shared_ptr<ngraph::Node> & node);

}  // namespace ngraph