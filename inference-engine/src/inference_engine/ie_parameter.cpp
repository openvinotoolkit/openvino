// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_parameter.hpp>
#include <memory>

#include <ngraph/variant.hpp>

namespace ngraph {

template class INFERENCE_ENGINE_API_CLASS(VariantImpl<InferenceEngine::Parameter>);
template <>
class INFERENCE_ENGINE_API_CLASS(VariantWrapper<InferenceEngine::Parameter>) : public VariantImpl<InferenceEngine::Parameter> {
public:
    static constexpr VariantTypeInfo type_info {"Variant::InferenceEngine::Parameter", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value): VariantImpl<value_type>(value) {}  // NOLINT
};

}  // namespace ngraph

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<InferenceEngine::Parameter>::type_info;

InferenceEngine::Parameter::Parameter(const std::shared_ptr<ngraph::Variant>& var) {
    if (auto paramWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<InferenceEngine::Parameter>>(var)) {
        auto param = paramWrapper->get();
        if (!param.empty()) ptr = param.ptr->copy();
    }
}

InferenceEngine::Parameter::Parameter(std::shared_ptr<ngraph::Variant>& var) {
    if (auto paramWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<InferenceEngine::Parameter>>(var)) {
        auto param = paramWrapper->get();
        if (!param.empty()) ptr = param.ptr->copy();
    }
}


std::shared_ptr<ngraph::Variant> InferenceEngine::Parameter::asVariant() const {
    return std::make_shared<ngraph::VariantWrapper<InferenceEngine::Parameter>>(*this);
}
