// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <bitset>
#include <ie_parameter.hpp>
#include <ie_context.hpp>
#include <ie_precision.hpp>
#include <ie_network.hpp>
#include <ie_preprocess.hpp>
#include <builders/ie_layer_builder.hpp>
#include <gna-api-types-xnn.h>
#include "ie_compound_blob.h"

#include "ie_blob.h"
#include <memory>

    template <class T>
    InferenceEngine::Precision  InferenceEngine::Precision::fromType(const char* typeName ) {
        return InferenceEngine::Precision(8 * sizeof(T), typeName == nullptr ? typeid(T).name() : typeName);
    }

template InferenceEngine::Precision InferenceEngine::Precision::fromType<short>(char const*);
template InferenceEngine::Precision InferenceEngine::Precision::fromType<int>(char const*);
template InferenceEngine::Precision InferenceEngine::Precision::fromType<signed char>(char const*);
template InferenceEngine::Precision InferenceEngine::Precision::fromType<_compound_bias_t>(char const*);
    template <class T>
    bool InferenceEngine::Precision::hasStorageType(const char* typeName ) const  noexcept{
        try {
            if (precisionInfo.value != BIN) {
                if (sizeof(T) != size()) {
                    std::cout << "hasStorageType T! size()" << std::endl;
                    return false;
                }
            }
#define CASE(x, y) \
    case x:        \
        return std::is_same<T, y>()
#define CASE2(x, y1, y2) \
    case x:              \
        return std::is_same<T, y1>() || std::is_same<T, y2>()

            switch (precisionInfo.value) {
                CASE(FP32, float);
                CASE2(FP16, int16_t, uint16_t);
                CASE(I16, int16_t);
                CASE(I32, int32_t);
                CASE(I64, int64_t);
                CASE(U16, uint16_t);
                CASE(U8, uint8_t);
                CASE(I8, int8_t);
                CASE2(Q78, int16_t, uint16_t);
                CASE2(BIN, int8_t, uint8_t);
            default:
                return areSameStrings(name(), typeName == nullptr ? (typeid(T).name()) : typeName);
#undef CASE
#undef CASE2
            }
        } catch (...) {
            return false;
        }
    }

template bool InferenceEngine::Precision::hasStorageType<unsigned short>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<unsigned char>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<short>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<float>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<signed char>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<int>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<long>(char const*) const;
template bool InferenceEngine::Precision::hasStorageType<_compound_bias_t>(char const*) const;




template <class T>
bool InferenceEngine::Parameter::is() const {
        return empty() ? false : ptr->is(typeid(T));;
}



 template <class T>
    bool InferenceEngine::Parameter::RealData<T>::is(const std::type_info& id) const {
    
    return id == typeid(T);
}
template <class T>
    InferenceEngine::Parameter::Any* InferenceEngine::Parameter::RealData<T>::copy() const  {
        return new RealData {get()};
    }
template <class T>
    T& InferenceEngine::Parameter::RealData<T>::get() & {
        return std::get<0>(*static_cast<std::tuple<T>*>(this));
    }
template <class T>
    const T& InferenceEngine::Parameter::RealData<T>::get() const& {
        return std::get<0>(*static_cast<const std::tuple<T>*>(this));
    }
template <class T>
template <class U>
        typename std::enable_if<!InferenceEngine::Parameter::HasOperatorEqual<U>::value, bool>::type InferenceEngine::Parameter::RealData<T>::equal(const InferenceEngine::Parameter::Any& left, const InferenceEngine::Parameter::Any& rhs) const {
            THROW_IE_EXCEPTION << "Parameter doesn't contain equal operator";
        }
template <class T>
        template <class U>
        typename std::enable_if<InferenceEngine::Parameter::HasOperatorEqual<U>::value, bool>::type InferenceEngine::Parameter::RealData<T>::equal(const InferenceEngine::Parameter::Any& left, const InferenceEngine::Parameter::Any& rhs) const {
            return InferenceEngine::Parameter::dyn_cast<U>(&left) == InferenceEngine::Parameter::dyn_cast<U>(&rhs);
        }
template <class T>
        bool InferenceEngine::Parameter::RealData<T>::operator==(const InferenceEngine::Parameter::Any& rhs) const {
            return equal<T>(*this, rhs);
        }

extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<int>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<bool>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<float>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<uint32_t>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<std::string>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<unsigned long>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<std::vector<int>>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<std::vector<std::string>>);
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<std::vector<unsigned long>>);
extern template struct INFERENCE_ENGINE_API_CLASS(
    InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>);
extern template struct INFERENCE_ENGINE_API_CLASS(
    InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>);
template struct InferenceEngine::Parameter::RealData<int>;
template struct InferenceEngine::Parameter::RealData<bool>;
template struct InferenceEngine::Parameter::RealData<float>;
template struct InferenceEngine::Parameter::RealData<uint32_t>;
template struct InferenceEngine::Parameter::RealData<std::string>;
template struct InferenceEngine::Parameter::RealData<unsigned long>;
template struct InferenceEngine::Parameter::RealData<std::vector<int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<bool>>;
template struct InferenceEngine::Parameter::RealData<std::__bit_reference<std::vector<bool>, true>>;
template struct InferenceEngine::Parameter::RealData<std::vector<char>>;
template struct InferenceEngine::Parameter::RealData<std::vector<std::string>>;
template struct InferenceEngine::Parameter::RealData<std::vector<unsigned long>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<float>>;
template struct InferenceEngine::Parameter::RealData<std::shared_ptr<InferenceEngine::Blob>>;
template struct InferenceEngine::Parameter::RealData<std::shared_ptr<InferenceEngine::Blob const>>;
template struct InferenceEngine::Parameter::RealData<std::shared_ptr<InferenceEngine::Builder::Layer>>;
template struct InferenceEngine::Parameter::RealData<std::vector<std::shared_ptr<InferenceEngine::Builder::Layer>>>;
template struct InferenceEngine::Parameter::RealData<InferenceEngine::Context>;
template struct InferenceEngine::Parameter::RealData<InferenceEngine::Connection>;
template struct InferenceEngine::Parameter::RealData<std::vector <InferenceEngine::Connection>>;
template struct InferenceEngine::Parameter::RealData<std::vector <InferenceEngine::Builder::Layer>>;
template struct InferenceEngine::Parameter::RealData<InferenceEngine::PreProcessInfo>;

template bool InferenceEngine::Parameter::is<int>() const;
// const;
template bool InferenceEngine::Parameter::is<bool>() const;
template bool InferenceEngine::Parameter::is<float>() const;

template bool InferenceEngine::Parameter::is<uint32_t>() const;

template bool InferenceEngine::Parameter::is<std::string>() const;

template bool InferenceEngine::Parameter::is<unsigned long>() const;    
//template bool InferenceEngine::Parameter::is<unsigned int>() const;
template bool InferenceEngine::Parameter::is<std::vector<int>>() const;
template bool InferenceEngine::Parameter::is<std::vector<unsigned int>>() const;
template bool InferenceEngine::Parameter::is<std::vector<bool>>() const;
template bool InferenceEngine::Parameter::is<std::__bit_reference<std::vector<bool>, true>>() const;

template bool InferenceEngine::Parameter::is<std::vector<float>>() const;
template bool InferenceEngine::Parameter::is<std::vector<std::string>>() const;

template bool InferenceEngine::Parameter::is<std::vector<unsigned long>>() const;

template bool InferenceEngine::Parameter::is<std::vector<InferenceEngine::Blob>>() const;
template bool InferenceEngine::Parameter::is<std::shared_ptr<InferenceEngine::Blob const>>() const;
template bool InferenceEngine::Parameter::is<std::shared_ptr<InferenceEngine::Blob>>() const;
template bool InferenceEngine::Parameter::is<std::vector<InferenceEngine::Port>>() const;
template bool InferenceEngine::Parameter::is<std::vector<InferenceEngine::PreProcessInfo>>() const;
template bool InferenceEngine::Parameter::is<InferenceEngine::PreProcessInfo>() const;
template bool InferenceEngine::Parameter::is<InferenceEngine::CompoundBlob>() const;

   // = (InferenceEngine::Parameter::RealData<std::vector<bool>>*)new InferenceEngine::Parameter::RealData<std::vector<bool>>();
//   new_vec_bool.push_back(true);
    //std::cout << new_vec_bool[0].is(typeid(std::vector<bool>));

// template bool InferenceEngine::Parameter::RealData<bool>::is(const std::type_info& id)const;
// template bool InferenceEngine::Parameter::RealData<float>::is(const std::type_info& id)const;
// template bool InferenceEngine::Parameter::RealData<uint32_t>::is(const std::type_info& id)const;
// template bool InferenceEngine::Parameter::RealData<std::string>::is(const std::type_info& id)const;
// template bool InferenceEngine::Parameter::RealData<unsigned long>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::vector<int>>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::vector<std::string>>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::vector<unsigned long>>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>::is(const std::type_info& id)const;
// template bool InferenceEngine::Parameter::RealData<std::vector<float>>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::shared_ptr<InferenceEngine::Blob>>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<InferenceEngine::Context>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<InferenceEngine::Connection>::is(const std::type_info& id) const;
// template bool InferenceEngine::Parameter::RealData<std::vector <InferenceEngine::Connection>>::is(const std::type_info& id) const;
#if defined(ENABLE_NGRAPH)

#include <ngraph/variant.hpp>

namespace ngraph {

template <>
class VariantWrapper<InferenceEngine::Parameter> : public VariantImpl<InferenceEngine::Parameter> {
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
#else
InferenceEngine::Parameter::Parameter(const std::shared_ptr<ngraph::Variant>& var) {}
InferenceEngine::Parameter::Parameter(std::shared_ptr<ngraph::Variant>& var) {}

std::shared_ptr<ngraph::Variant> InferenceEngine::Parameter::asVariant() const {
    return nullptr;
}
#endif
