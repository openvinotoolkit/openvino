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
#include <transformations_visibility.hpp>

namespace ngraph {

class TRANSFORMATIONS_API PrimitivesPriority {
private:
    std::string primitives_priority;

public:
    PrimitivesPriority() = default;

    explicit PrimitivesPriority(const std::string &primitives_priority) : primitives_priority(primitives_priority) {}

    std::string getPrimitivesPriority() const;
};

template<>
class TRANSFORMATIONS_API VariantWrapper<PrimitivesPriority> : public VariantImpl<PrimitivesPriority> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::PrimitivesPriority", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;
};

TRANSFORMATIONS_API std::string getPrimitivesPriority(const std::shared_ptr<ngraph::Node> & node);

}  // namespace ngraph
