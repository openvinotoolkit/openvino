// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/error.hpp>

#include <memory>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>
#include <typeinfo>

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

        explicit HolderImpl(const T& val) :
                _val(val) {
        }
        explicit HolderImpl(T&& val) :
                _val(std::move(val)) {
        }

        void set(const T& val) {
            _val = val;
        }
        void set(T&& val) {
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
    Any() = default;

    Any(Any&&) = default;
    Any& operator=(Any&&) = default;

    Any(const Any& other) :
            _impl(other._impl != nullptr ? other._impl->clone() : nullptr) {
    }
    Any& operator=(const Any& other) {
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
        typename = typename std::enable_if<!std::is_reference<T>::value, void>::type>
    explicit Any(T&& arg) :
            _impl(new HolderImpl<typename std::decay<T>::type>(std::move(arg))) {
    }

    bool empty() const {
        return _impl == nullptr;
    }

    template <typename T>
    void set(const T& arg) {
        if (auto casted = dynamic_cast<HolderImpl<T>*>(_impl.get())) {
            casted->set(arg);
        } else {
            _impl.reset(new HolderImpl<T>(arg));
        }
    }
    template <
        typename T,
        typename = typename std::enable_if<!std::is_reference<T>::value, void>::type>
    void set(T&& arg) {
        if (auto casted = dynamic_cast<HolderImpl<typename std::decay<T>::type>*>(_impl.get())) {
            casted->set(std::move(arg));
        } else {
            _impl.reset(new HolderImpl<T>(std::move(arg)));
        }
    }

    template <typename T>
    const T& get() const {
        VPU_INTERNAL_CHECK(_impl != nullptr, "Any object was not set");
        auto casted = dynamic_cast<const HolderImpl<T>*>(_impl.get());
        VPU_INTERNAL_CHECK(casted != nullptr, "Any object has type different than %v", typeid(T).name());
        return casted->_val;
    }
    template <typename T>
    T& get() {
        VPU_INTERNAL_CHECK(_impl != nullptr, "Any object was not set");
        auto casted = dynamic_cast<HolderImpl<T>*>(_impl.get());
        VPU_INTERNAL_CHECK(casted != nullptr, "Any object has type different than %v", typeid(T).name());
        return casted->_val;
    }

    void swap(Any& other) {
        std::swap(_impl, other._impl);
    }

    template <class Out>
    void printImpl(Out& out) const {
        if (!empty()) {
            _impl->printImpl(out);
        }
    }

private:
    Holder::Ptr _impl;
};

inline void printTo(std::ostream& os, const Any& any) {
    any.printImpl(os);
}
inline void printTo(DotLabel& lbl, const Any& any) {
    any.printImpl(lbl);
}

}  // namespace vpu
