// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the AliasMap class
 *
 * @file ie_alias_map.hpp
 */

#pragma once

#include <map>
#include <unordered_set>

namespace InferenceEngine {
namespace details {

template<
    class Key,
    class T,
    class Compare = std::less<Key>,
    class Allocator = std::allocator<std::pair<const Key, T> >
> class AliasMap {
private:
    std::map<Key, T, Compare, Allocator> mmap;

public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const key_type, mapped_type>;
    using key_compare = typename std::map<Key, T, Compare, Allocator>::key_compare;
    using allocator_type = typename std::map<Key, T, Compare, Allocator>::allocator_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = typename std::map<Key, T, Compare, Allocator>::iterator;
    using const_iterator = typename std::map<Key, T, Compare, Allocator>::const_iterator;
    using reverse_iterator = typename std::map<Key, T, Compare, Allocator>::reverse_iterator;
    using const_reverse_iterator = typename std::map<Key, T, Compare, Allocator>::const_reverse_iterator;

    AliasMap() = default;
    explicit AliasMap(const key_compare& comp): mmap(comp) {}
    explicit AliasMap(const key_compare& comp, const allocator_type& type): mmap(comp, type) {}
    explicit AliasMap(std::initializer_list<value_type> list, const key_compare& comp = key_compare()): mmap(list, comp) {}
    explicit AliasMap(std::initializer_list<value_type> list, const key_compare& comp, const allocator_type& a): mmap(list, comp, a) {}
    explicit AliasMap(std::initializer_list<value_type> list, const allocator_type& a): mmap(list, a) {}
    AliasMap(const std::map<Key, T, Compare, Allocator>& map): mmap(map) {}
    AliasMap(std::map<Key, T, Compare, Allocator>&& map) noexcept: mmap(std::move(map)) {}
    AliasMap(const AliasMap& map): mmap(map.mmap) {}
    AliasMap(AliasMap&& map) noexcept: mmap(std::move(map.mmap)) {}
    virtual ~AliasMap() = default;

    AliasMap& operator=(std::initializer_list<value_type> list) {
        this->mmap = list;
        return *this;
    }

    AliasMap& operator=(const std::map<Key, T, Compare, Allocator>& map) {
        this->mmap = map;
        return *this;
    }
    AliasMap& operator=(std::map<Key, T, Compare, Allocator>&& map) noexcept {
        this->mmap = std::move(map.mmap);
        return *this;
    }
    AliasMap& operator=(const AliasMap& map) {
        this->mmap = map.mmap;
        return *this;
    }
    AliasMap& operator=(AliasMap&& map) noexcept {
        this->mmap = std::move(map.mmap);
        return *this;
    }

    operator const std::map<Key, T, Compare, Allocator>&() const {
        return mmap;
    }

    operator std::map<Key, T, Compare, Allocator>&() {
        return mmap;
    }

    iterator find(const Key& k) {
        return mmap.find(k);
    }

    const_iterator find(const Key& k) const {
        return mmap.find(k);
    }

    iterator begin() {
        return mmap.begin();
    }

    const_iterator begin() const {
        return mmap.begin();
    }

    iterator end() {
        return mmap.end();
    }

    const_iterator end() const {
        return mmap.end();
    }

    reverse_iterator rbegin() {
        return mmap.rbegin();
    }

    const_reverse_iterator rbegin() const {
        return mmap.rbegin();
    }

    reverse_iterator rend() {
        return mmap.rend();
    }

    const_reverse_iterator rend() const {
        return mmap.rend();
    }

    const_iterator cbegin() const {
        return begin();
    }

    const_iterator cend() const {
        return end();
    }

    const_reverse_iterator crbegin() const {
        return rbegin();
    }

    const_reverse_iterator crend() const {
        return rend();
    }

    size_t size() const {
        size_t real_size(0);
        std::unordered_set<T> values;
        for (const auto& val : mmap) {
            if (values.find(val.second) == values.end()) {
                real_size++;
                values.insert(val.second);
            }
        }
        return real_size;
    }

    size_t max_size() const { return mmap.max_size(); }

    bool empty() const { return mmap.empty(); }

    T& operator[](const Key& k) { return mmap[k]; }

    T& operator[](Key&& k) { return mmap[k]; }

    T& at(const Key& k) { return mmap.at(k); }

    const T& at(const Key& k) const { return mmap.at(k); }

    iterator erase(const_iterator p) { return mmap.erase(p); }
    iterator erase(iterator p) { return mmap.erase(p); }
    typename std::map<Key, T, Compare, Allocator>::size_type erase(const Key& k) { return mmap.erase(k); }
    iterator erase(const_iterator f, const_iterator l) { return mmap.erase(f, l); }

    void clear() noexcept { mmap.clear(); }

    template <class ..._Args>
    std::pair<iterator, bool> emplace(_Args&& ...__args) {
        return mmap.emplace(std::forward<_Args>(__args)...);
    }
    typename std::map<Key, T, Compare, Allocator>::size_type count(const Key& k) { return mmap.count(k); }

    bool operator==(const std::map<Key, T, Compare, Allocator>& other) {
        return mmap == other;
    }

    bool operator!=(const std::map<Key, T, Compare, Allocator>& other) {
        return mmap != other;
    }

    bool operator<(const std::map<Key, T, Compare, Allocator>& other) {
        return mmap < other;
    }

    bool operator>(const std::map<Key, T, Compare, Allocator>& other) {
        return mmap > other;
    }

    bool operator<=(const std::map<Key, T, Compare, Allocator>& other) {
        return mmap <= other;
    }

    bool operator>=(const std::map<Key, T, Compare, Allocator>& other) {
        return mmap >= other;
    }
};

template <class Key, class T, class Compare, class Allocator>
bool
operator==(const AliasMap<Key, T, Compare, Allocator>& x,
           const std::map<Key, T, Compare, Allocator>& y)
{
    return static_cast<const std::map<Key, T, Compare, Allocator>>(x) == y;
}

template <class Key, class T, class Compare, class Allocator>
bool
operator==(const AliasMap<Key, T, Compare, Allocator>& x,
           const AliasMap<Key, T, Compare, Allocator>& y)
{
    return x == static_cast<const std::map<Key, T, Compare, Allocator>>(y);
}

template <class Key, class T, class Compare, class Allocator>
bool
operator!=(const AliasMap<Key, T, Compare, Allocator>& x,
           const std::map<Key, T, Compare, Allocator>& y)
{
    return static_cast<const std::map<Key, T, Compare, Allocator>>(x) != y;
}

template <class Key, class T, class Compare, class Allocator>
bool
operator!=(const AliasMap<Key, T, Compare, Allocator>& x,
           const AliasMap<Key, T, Compare, Allocator>& y)
{
    return x != static_cast<const std::map<Key, T, Compare, Allocator>>(y);
}

template <class Key, class T, class Compare, class Allocator>
bool
operator<(const AliasMap<Key, T, Compare, Allocator>& x,
          const std::map<Key, T, Compare, Allocator>& y)
{
    return static_cast<const std::map<Key, T, Compare, Allocator>>(x) < y;
}

template <class Key, class T, class Compare, class Allocator>
bool
operator<(const AliasMap<Key, T, Compare, Allocator>& x,
           const AliasMap<Key, T, Compare, Allocator>& y)
{
    return x < static_cast<const std::map<Key, T, Compare, Allocator>>(y);
}

template <class Key, class T, class Compare, class Allocator>
bool
operator<=(const AliasMap<Key, T, Compare, Allocator>& x,
           const std::map<Key, T, Compare, Allocator>& y)
{
    return static_cast<const std::map<Key, T, Compare, Allocator>>(x) <= y;
}

template <class Key, class T, class Compare, class Allocator>
bool
operator<=(const AliasMap<Key, T, Compare, Allocator>& x,
           const AliasMap<Key, T, Compare, Allocator>& y)
{
    return x <= static_cast<const std::map<Key, T, Compare, Allocator>>(y);
}

template <class Key, class T, class Compare, class Allocator>
bool
operator>(const AliasMap<Key, T, Compare, Allocator>& x,
          const std::map<Key, T, Compare, Allocator>& y)
{
    return static_cast<const std::map<Key, T, Compare, Allocator>>(x) > y;
}

template <class Key, class T, class Compare, class Allocator>
bool
operator>(const AliasMap<Key, T, Compare, Allocator>& x,
           const AliasMap<Key, T, Compare, Allocator>& y)
{
    return x > static_cast<const std::map<Key, T, Compare, Allocator>>(y);
}

template <class Key, class T, class Compare, class Allocator>
bool
operator>=(const AliasMap<Key, T, Compare, Allocator>& x,
           const std::map<Key, T, Compare, Allocator>& y)
{
    return static_cast<const std::map<Key, T, Compare, Allocator>>(x) >= y;
}

template <class Key, class T, class Compare, class Allocator>
bool
operator>=(const AliasMap<Key, T, Compare, Allocator>& x,
           const AliasMap<Key, T, Compare, Allocator>& y)
{
    return x >= static_cast<const std::map<Key, T, Compare, Allocator>>(y);
}

}  // namespace details
}  // namespace InferenceEngine
