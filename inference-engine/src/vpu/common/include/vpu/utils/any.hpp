// Copyright (C) 2018-2020 Intel Corporation
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
        T _val;

        explicit inline HolderImpl(const T& val) :
                _val(val) {
        }
        explicit inline HolderImpl(T&& val) :
                _val(std::move(val)) {
        }

        inline void set(const T& val) {
            _val = val;
        }
        inline void set(T&& val) {
            _val = std::move(val);
        }

        ~HolderImpl() override = default;

        Holder::Ptr clone() const override {
            return Holder::Ptr(new HolderImpl<T>(_val));
        }

        void printImpl(std::ostream& os) const override {
            printTo(os, _val);
        }
        void printImpl(DotLabel& lbl) const override {
            printTo(lbl, _val);
        }
    };

public:
    inline Any() = default;

    inline Any(Any&&) = default;
    inline Any& operator=(Any&&) = default;

    inline Any(const Any& other) :
            _impl(other._impl != nullptr ? other._impl->clone() : nullptr) {
    }
    inline Any& operator=(const Any& other) {
        if (&other != this) {
            if (other._impl == nullptr) {
                _impl.reset();
            } else {
                _impl = other._impl->clone();
            }
        }
        return *this;
    }

    template <typename T>
    explicit Any(const T& arg) :
            _impl(new HolderImpl<T>(arg)) {
    }
    template <
        typename T,
        typename _Check = typename std::enable_if<!std::is_reference<T>::value, void>::type>
    explicit inline Any(T&& arg) :
            _impl(new HolderImpl<typename std::decay<T>::type>(std::move(arg))) {
    }

    inline bool empty() const {
        return _impl == nullptr;
    }

    template <typename T>
    inline void set(const T& arg) {
        if (auto casted = dynamic_cast<HolderImpl<T>*>(_impl.get())) {
            casted->set(arg);
        } else {
            _impl.reset(new HolderImpl<T>(arg));
        }
    }
    template <
        typename T,
        typename _Check = typename std::enable_if<!std::is_reference<T>::value, void>::type>
    inline void set(T&& arg) {
        if (auto casted = dynamic_cast<HolderImpl<typename std::decay<T>::type>*>(_impl.get())) {
            casted->set(std::move(arg));
        } else {
            _impl.reset(new HolderImpl<T>(std::move(arg)));
        }
    }

    template <typename T>
    inline const T& get() const {
        auto casted = dynamic_cast<const HolderImpl<T>*>(_impl.get());
        IE_ASSERT(casted != nullptr);
        return casted->_val;
    }
    template <typename T>
    inline T& get() {
        auto casted = dynamic_cast<HolderImpl<T>*>(_impl.get());
        IE_ASSERT(casted != nullptr);
        return casted->_val;
    }

    inline void swap(Any& other) {
        std::swap(_impl, other._impl);
    }

    inline void printImpl(std::ostream& os) const {
        if (_impl != nullptr) {
            _impl->printImpl(os);
        }
    }

    inline void printImpl(DotLabel& lbl) const {
        if (_impl != nullptr) {
            _impl->printImpl(lbl);
        }
    }

private:
    Holder::Ptr _impl;
};

}  // namespace vpu
