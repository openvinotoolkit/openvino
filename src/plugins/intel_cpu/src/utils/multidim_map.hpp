// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <unordered_map>
#include <type_traits>
#include <functional>

namespace ov {
namespace intel_cpu {

namespace internal {

template <typename K>
struct enum_hash {
    std::size_t operator()(K t) const {
        return static_cast<std::size_t>(t);
    }
};

template<typename K>
using hash_t = typename std::conditional<std::is_enum<K>::value, enum_hash<K>, std::hash<K>>::type;

}   // namespace internal

template<typename K, typename ...Ts>
struct multidim_map {
    using key_type = K;
    using mapped_type = multidim_map<Ts...>;
    using hash_type = internal::hash_t<K>;

public:
    mapped_type & operator[](const key_type & key) {
        return _map[key];
    }

    const mapped_type & at(const key_type & key) const {
        return _map.at(key);
    }

private:
    std::unordered_map<key_type, mapped_type, hash_type> _map;
};

template<typename K, typename T>
struct multidim_map<K, T> {
    using key_type = K;
    using mapped_type = T;
    using hash_type = internal::hash_t<K>;

public:
    mapped_type & operator[](const key_type & key) {
        return _map[key];
    }

    const mapped_type & at(const key_type & key) const {
        return _map.at(key);
    }

private:
    std::unordered_map<key_type, mapped_type, hash_type> _map;
};

}   // namespace intel_cpu
}   // namespace ov
