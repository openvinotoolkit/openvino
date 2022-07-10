// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <memory>

namespace ov {
namespace frontend {

/**
 * \brief Extension to Derived class by adding shared object smart pointer.
 *
 * Class provide helper methods to make smart pointers from  pointer return by pimpl and shared object which can
 * provide correct life time of required shared object (library).
 *
 * Rationale:
 *
 * To correct use the pimpl pattern with dynamically loaded libraries the class which has got the pimpl has to have
 * pointer to library which is required by pimpl. All methods which use return values (smart pointers) from pimpl
 * cannot return them directly. These pointers and shared object has to be used as input data for new smart pointers
 * for type which has got pimpl member.
 *
 * Smart pointer returned directly from pimpl can ensure enough life time for library to destroy object pointed by
 * smart pointer but will fail on destroy of smart pointer object.
 *
 * \tparam Derived  Class which get features from SharedObjectExtension.
 */
template <class Derived>
class SharedObjectExtension {
protected:
    std::shared_ptr<void> m_shared_object;

public:
    SharedObjectExtension() = default;  //!< Required by class which got only default ctor.

    /**
     * \brief Construct a new Shared Object Extension object.
     *
     * \param shared_object  Smart pointer to library object.
     */
    explicit SharedObjectExtension(std::shared_ptr<void> shared_object) : m_shared_object{std::move(shared_object)} {}

    /**
     * \brief Make shared pointer of T class from TPimpl and shared object (library).
     *
     * Shared object is library pointer required to prevent close library before pimpl destruction.
     *
     * The function will be enabled if T class can be constructed from TPimpl and m_shared_object types.
     *
     * \tparam T       Type pointed by shared pointer.
     * \tparam TPimpl  Type of pimpl for T class which depends on shared object (library).
     *
     * \param pimpl    Instance of implementation for T class.
     *
     * \return std::shared_ptr<T>  Shared pointer of T class or null if pimpl is nullptr.
     */
    template <
        class T,
        class TPimpl,
        typename std::enable_if<std::is_constructible<T, TPimpl, decltype(m_shared_object)>::value, bool>::type = true>
    std::shared_ptr<T> make_shared_w_pimpl(const TPimpl& pimpl) const {
        if (pimpl) {
            return std::make_shared<T>(pimpl, m_shared_object);
        }
        return {};
    };

    /**
     * \brief Make shared pointer of Derived class from TPimpl and shared object (library).
     *
     * \tparam TPimpl  Type of pimpl for Derived class which depends on shared object (library).
     *
     * \param pimpl    Instance of implementation for Derived class.
     *
     * \return std::shared_ptr<Derived>  Shared pointer of Derived class or null if pimpl is nullptr.
     */
    template <class TPimpl>
    std::shared_ptr<Derived> make_shared_w_pimpl(const TPimpl& pimpl) const {
        return make_shared_w_pimpl<Derived>(pimpl);
    };

    /**
     * \brief Transform input container of pimpls into same type output containers but elements are shared pointers
     * of T class made from input pimpls and shared_object.
     *
     * T class has to be convertible into type of elemets of input container.
     *
     * \tparam T          Type of ouput shared pointers.
     * \tparam Container  Type of input container.
     * \tparam DType      Type of container element.
     *
     * \param in          Input container of pimpls.
     *
     * \return Container of shared pointers of T type.
     */
    template <class T,
              class Container,
              class DType = typename Container::value_type,
              typename std::enable_if<std::is_convertible<std::shared_ptr<T>, DType>::value, bool>::type = true>
    Container transform_pimpls(const Container& in) const {
        Container out;
        out.reserve(in.size());

        std::transform(in.begin(), in.end(), std::back_inserter(out), [this](const DType& el) {
            return make_shared_w_pimpl<T>(el);
        });

        return out;
    }

    /**
     * \brief Transform input container of pimpls into same type output containers but elements are shared pointers
     * of Derived class made from input pimpls and shared_object.
     *
     * \tparam Container  Type of input containers im pimpls.
     *
     * \param in          Input container of pimpls.
     *
     * \return Container of shared pointers of Derived type.
     */
    template <class Container>
    Container transform_pimpls(const Container& in) const {
        return transform_pimpls<Derived>(in);
    }
};
}  // namespace frontend
}  // namespace ov
