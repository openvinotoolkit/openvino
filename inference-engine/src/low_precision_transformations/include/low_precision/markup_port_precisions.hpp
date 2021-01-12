// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API MarkupPortPrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// TODO: move to inside Key
//Restrictions
using PrecisionsByPort = std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>;

class OperationRestriction {
public:
    std::string name;
    uint64_t version;
    std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>> precisionsByPort;

    OperationRestriction() = default;
    OperationRestriction(
        const std::string& name,
        const uint64_t version,
        const PrecisionsByPort& precisionsByPort) :
        name(name), version(version), precisionsByPort(precisionsByPort) {}

    //template <typename T>
    //static std::pair<std::string, std::vector<uint64_t>> create(const size_t port, const bool specifiedVersion = false) {
    //    const ngraph::Node::type_info_t& typeInfo = T::get_type_info_static();
    //    return { typeInfo.name, specifiedVersion ? std::vector<uint64_t>{typeInfo.version} : std::vector<uint64_t>{} };
    //}

    template <typename T>
    static OperationRestriction create(
        const PrecisionsByPort& precisionsByPort,
        const bool specifiedVersion = false) {
        const ngraph::Node::type_info_t& typeInfo = T::get_type_info_static();
        return OperationRestriction(typeInfo.name, specifiedVersion ? typeInfo.version : -1ull, precisionsByPort);
    }
};

//class RestrictionAttribute {
//public:
//    RestrictionAttribute(const std::vector<ngraph::element::Type>& precisions) : precisions(precisions) {};
//    std::vector<ngraph::element::Type> precisions;
//};

using RestrictionAttribute = std::vector<ngraph::element::Type>;

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<RestrictionAttribute>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<RestrictionAttribute> : public ngraph::VariantImpl<RestrictionAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "RestrictionAttribute", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;
};



// Transformation is used to add customization options runtime
class ngraph::pass::low_precision::MarkupPortPrecisions : public ngraph::pass::FunctionPass {
public:
    class Restriction {
    public:
        Restriction() = default;
        Restriction(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const uint64_t version, const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>& precisions) {
            precisionsByVersion.emplace(version, precisions);
        }

        bool versionIsRequired;
        std::map<uint64_t, std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>> precisionsByVersion;
    };

    MarkupPortPrecisions(const std::vector<OperationRestriction>& restrictions);
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    std::unordered_map<std::string, Restriction> restrictionsByOperation;
};
