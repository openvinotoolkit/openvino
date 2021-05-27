// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Parameter class
 * @file ie_parameter.hpp
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include "ie_blob.h"

namespace ngraph {

class Variant;

}  // namespace ngraph

namespace InferenceEngine {

/**
 * @brief This class represents an object to work with different parameters
 *
 */
class INFERENCE_ENGINE_API_CLASS(Parameter) {
public:
    /**
     * @brief Default constructor
     */
    Parameter() = default;

    /**
     * @brief Move constructor
     *
     * @param parameter Parameter object
     */
    Parameter(Parameter&& parameter) noexcept {
        std::swap(ptr, parameter.ptr);
    }

    /**
     * @deprecated Use ngraph::Variant directly
     * @brief Creates parameter from variant.
     * This method creates empty parameter if variant doesn't contain Parameter
     *
     * @param var ngraph variant
     */
    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Variant directly")
    Parameter(const std::shared_ptr<ngraph::Variant>& var);

    /**
     * @deprecated Use ngraph::Variant directly
     * @brief Creates parameter from variant.
     * This method creates empty parameter if variant doesn't contain Parameter
     *
     * @param var ngraph variant
     */
    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Variant directly")
    Parameter(std::shared_ptr<ngraph::Variant>& var);

    /**
     * @brief Copy constructor
     *
     * @param parameter Parameter object
     */
    Parameter(const Parameter& parameter) {
        *this = parameter;
    }

    /**
     * @brief Constructor creates parameter with object
     *
     * @tparam T Parameter type
     * @tparam U Identity type-transformation
     * @param parameter object
     */
    template <class T,
              typename = typename std::enable_if<!std::is_same<typename std::decay<T>::type, Parameter>::value>::type>
    Parameter(T&& parameter) {  // NOLINT
        static_assert(!std::is_same<typename std::decay<T>::type, Parameter>::value, "To prevent recursion");
        ptr = new RealData<typename std::decay<T>::type>(std::forward<T>(parameter));
    }

    /**
     * @brief Constructor creates string parameter from char *
     *
     * @param str char array
     */
    Parameter(const char* str): Parameter(std::string(str)) {}  // NOLINT

    /**
     * @brief Destructor
     */
    virtual ~Parameter();

    /**
     * Copy operator for Parameter
     * @param parameter Parameter object
     * @return Parameter
     */
    Parameter& operator=(const Parameter& parameter) {
        if (this == &parameter) {
            return *this;
        }
        clear();
        if (!parameter.empty()) ptr = parameter.ptr->copy();
        return *this;
    }

    /**
     * Remove a value from parameter
     */
    void clear() {
        delete ptr;
        ptr = nullptr;
    }

    /**
     * Checks that parameter contains a value
     * @return false if parameter contains a value else false
     */
    bool empty() const noexcept {
        return nullptr == ptr;
    }

    /**
     * Checks the type of value
     * @tparam T Type of value
     * @return true if type of value is correct
     */
    template <class T>
    bool is() const {
        return empty() ? false : ptr->is(typeid(T));
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    T&& as() && {
        return std::move(dyn_cast<T>(ptr));
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    T& as() & {
        return dyn_cast<T>(ptr);
    }
    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    const T& as() const& {
        return dyn_cast<T>(ptr);
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    operator T &&() && {
        return std::move(dyn_cast<typename std::remove_cv<T>::type>(ptr));
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    operator T&() & {
        return dyn_cast<typename std::remove_cv<T>::type>(ptr);
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    operator const T&() const& {
        return dyn_cast<typename std::remove_cv<T>::type>(ptr);
    }

    /**
     * @deprecated Use ngraph::Variant directly
     * @brief Converts parameter to shared pointer on ngraph::Variant
     *
     * @return shared pointer on ngraph::Variant
     */
    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Variant directly")
    std::shared_ptr<ngraph::Variant> asVariant() const;

    /**
     * @deprecated Use ngraph::Variant directly
     * @brief Casts to shared pointer on ngraph::Variant
     *
     * @return shared pointer on ngraph::Variant
     */
    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Variant directly")
    operator std::shared_ptr<ngraph::Variant>() const {
        IE_SUPPRESS_DEPRECATED_START
        return asVariant();
        IE_SUPPRESS_DEPRECATED_END
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    operator T&() const& {
        return dyn_cast<typename std::remove_cv<T>::type>(ptr);
    }

    /**
     * @brief The comparison operator for the Parameter
     *
     * @param rhs object to compare
     * @return true if objects are equal
     */
    bool operator==(const Parameter& rhs) const {
        return *ptr == *(rhs.ptr);
    }
    /**
     * @brief The comparison operator for the Parameter
     *
     * @param rhs object to compare
     * @return true if objects aren't equal
     */
    bool operator!=(const Parameter& rhs) const {
        return !(*this == rhs);
    }

private:
    template <class T, class EqualTo>
    struct CheckOperatorEqual {
        template <class U, class V>
        static auto test(U*) -> decltype(std::declval<U>() == std::declval<V>()) {
            return false;
        }

        template <typename, typename>
        static auto test(...) -> std::false_type {
            return {};
        }

        using type = typename std::is_same<bool, decltype(test<T, EqualTo>(nullptr))>::type;
    };

    template <class T, class EqualTo = T>
    struct HasOperatorEqual : CheckOperatorEqual<T, EqualTo>::type {};

    struct Any {
#ifdef __ANDROID__
        virtual ~Any();
#else
        virtual ~Any() = default;
#endif
        virtual bool is(const std::type_info&) const = 0;
        virtual Any* copy() const = 0;
        virtual bool operator==(const Any& rhs) const = 0;
    };

    template <class T>
    struct RealData : Any, std::tuple<T> {
        using std::tuple<T>::tuple;

        bool is(const std::type_info& id) const override {
            return id == typeid(T);
        }
        Any* copy() const override {
            return new RealData {get()};
        }

        T& get() & {
            return std::get<0>(*static_cast<std::tuple<T>*>(this));
        }

        const T& get() const& {
            return std::get<0>(*static_cast<const std::tuple<T>*>(this));
        }

        template <class U>
        typename std::enable_if<!HasOperatorEqual<U>::value, bool>::type
        equal(const Any& left, const Any& rhs) const {
            IE_THROW() << "Parameter doesn't contain equal operator";
        }

        template <class U>
        typename std::enable_if<HasOperatorEqual<U>::value, bool>::type
        equal(const Any& left, const Any& rhs) const {
            return dyn_cast<U>(&left) == dyn_cast<U>(&rhs);
        }

        bool operator==(const Any& rhs) const override {
            return rhs.is(typeid(T)) && equal<T>(*this, rhs);
        }
    };

    template <typename T>
    static T& dyn_cast(Any* obj) {
        if (obj == nullptr) IE_THROW() << "Parameter is empty!";
        return dynamic_cast<RealData<T>&>(*obj).get();
    }

    template <typename T>
    static const T& dyn_cast(const Any* obj) {
        if (obj == nullptr) IE_THROW() << "Parameter is empty!";
        return dynamic_cast<const RealData<T>&>(*obj).get();
    }

    Any* ptr = nullptr;
};

/**
 * @brief An std::map object containing parameters
  */
using ParamMap = std::map<std::string, Parameter>;

#ifdef __ANDROID__
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<InferenceEngine::Blob::Ptr>);
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
#endif

}  // namespace InferenceEngine
