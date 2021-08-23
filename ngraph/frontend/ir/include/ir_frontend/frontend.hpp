// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>
#include <inference_engine.hpp>
#include <ngraph/variant.hpp>
#include <pugixml.hpp>

#include "utility.hpp"

namespace ngraph {
namespace frontend {

class IR_API FrontEndIR : public FrontEnd {
public:
    FrontEndIR() = default;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted nGraph function
    /// \return fully converted nGraph function
    std::shared_ptr<Function> convert(InputModel::Ptr model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return IR frontend name.
    std::string get_name() const override;

protected:
    /// \brief Check if FrontEndPDPD can recognize model from given parts
    /// \param params Can be path to folder which contains __model__ file or path to
    /// .pdmodel file
    /// \return InputModel::Ptr
    bool supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override;

    /// \brief Reads model from 1 or 2 given file names or 1 or 2 std::istream containing
    /// model in protobuf format and weights
    /// \param params Can contain path to folder with __model__ file or path to .pdmodel
    /// file or 1 or 2 streams with model and weights
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override;
};

}  // namespace frontend
}  // namespace ngraph

namespace ov {

template <>
class IR_API VariantWrapper<pugi::xml_node> : public VariantImpl<pugi::xml_node> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::pugi::xml_node", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

template <>
class IR_API VariantWrapper<InferenceEngine::Blob::CPtr> : public VariantImpl<InferenceEngine::Blob::CPtr> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Blob::CPtr", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

template <>
class IR_API VariantWrapper<std::vector<InferenceEngine::IExtensionPtr>>
    : public VariantImpl<std::vector<InferenceEngine::IExtensionPtr>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Extensions", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

}  // namespace ov
