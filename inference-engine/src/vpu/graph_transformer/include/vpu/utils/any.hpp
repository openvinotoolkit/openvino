// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>

#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>

namespace vpu {

class Any final {
    struct Holder {
        using Ptr = std::unique_ptr<Holder>;

        virtual ~Holder() = default;

        virtual Holder::Ptr clone() const = 0;

        virtual void printImpl(std::ostream& os) const = 0;
        virtual void printImpl(DotLabel& lbl) const = 0;
    };

    template <typename T>
    struct HolderImpl final : Holder {
        T val;

        template <typename U>
        explicit HolderImpl(U&& val) : val(std::forward<U>(val)) {}

        Holder::Ptr clone() const override { return Holder::Ptr(new HolderImpl(val)); }

        void printImpl(std::ostream& os) const override { printTo(os, val); }
        void printImpl(DotLabel& lbl) const override { printTo(lbl, val); }
    };

public:
    Any() = default;
    Any(Any&&) = default;
    Any& operator=(Any&&) = default;

    template <typename T>
    explicit Any(T&& arg) : _impl(new HolderImpl<typename std::decay<T>::type>(std::forward<T>(arg))) {}

    Any(const Any& other) : _impl(other._impl != nullptr ? other._impl->clone() : nullptr) {}

    Any& operator=(const Any& other) {
        Any temp(other);
        swap(temp);
        return *this;
    }

    void swap(Any& other) {
        std::swap(_impl, other._impl);
    }

    template <typename T>
    const T& cast() const {
        auto casted = dynamic_cast<const HolderImpl<typename std::decay<T>::type>*>(_impl.get());
        IE_ASSERT(casted != nullptr);
        return casted->val;
    }

    template <typename T>
    T& cast() {
        auto casted = dynamic_cast<HolderImpl<typename std::decay<T>::type>*>(_impl.get());
        IE_ASSERT(casted != nullptr);
        return casted->val;
    }

    void printImpl(std::ostream& os) const {
        if (_impl != nullptr)
            _impl->printImpl(os);
    }

    void printImpl(DotLabel& lbl) const {
        if (_impl != nullptr)
            _impl->printImpl(lbl);
    }

private:
    Holder::Ptr _impl;
};

}  // namespace vpu
