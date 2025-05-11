// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include <utility>
#include <iostream>

namespace cldnn {

template <typename T, typename Enable = void>
class static_instance;

template <typename T>
class static_instance<T, typename std::enable_if<std::is_default_constructible<T>::value>::type> {
public:
    static T& get_instance() {
        return instantiate();
    }

private:
    static T& instantiate() {
        static T singleton;
        (void)instance;
        return singleton;
    }

    static const T& instance;
};

template <typename T>
const T& static_instance<T, typename std::enable_if<std::is_default_constructible<T>::value>::type>::instance = static_instance<T>::instantiate();

template <typename T>
class static_instance<T, typename std::enable_if<!std::is_default_constructible<T>::value>::type> {
public:
    static T& get_instance() {
        return instantiate();
    }

private:
    static T& instantiate() {
        (void)instance;
        return T::instance();
    }

    static const T& instance;
};

template <typename T>
const T& static_instance<T, typename std::enable_if<!std::is_default_constructible<T>::value>::type>::instance = static_instance<T>::instantiate();

}  // namespace cldnn
