// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <unordered_map>
#include <vector>

namespace ov {
namespace intel_cpu {

template<typename Key, typename Val>
class ordered_map {
public:
    using key_type = Key;
    using value_type = std::pair<Key, Val>;
    using mapped_type = Val;
    using sequence_type = std::vector<value_type>;

    using iterator = typename sequence_type::iterator;
    using const_iterator = typename sequence_type::const_iterator;

public:
    ordered_map() = default;

    std::pair<iterator, bool> insert(const value_type& value) {
        auto itr = m_map.find(value.first);
        if (itr != m_map.end()) {
            return {m_vec.begin() + itr->second, false};
        }
        m_map.insert({value.first, m_vec.size()});
        m_vec.push_back(value);
        return {--m_vec.end(), true};
    }

    std::pair<iterator, bool> insert(value_type&& value) {
        auto itr = m_map.find(value.first);
        if (itr != m_map.end()) {
            return {m_vec.begin() + itr->second, false};
        }
        m_map.insert({value.first, m_vec.size()});
        m_vec.push_back(std::move(value));
        return {--m_vec.end(), true};
    }

    mapped_type& operator[](const key_type& key) {
        auto itr = m_map.find(key);
        if (itr != m_map.end()) {
            return m_vec[itr->second];
        }
        m_map.insert({key, m_vec.size()});
        m_vec.emplace_back();
        return m_vec.back().second;
    }

    iterator begin() {
        return m_vec.begin();
    }

    iterator end() {
        return m_vec.end();
    }

    bool empty() const {
        return m_vec.empty();
    }

    size_t count(const key_type& key) const {
        return m_map.count(key);
    }

    iterator find(const key_type& key) {
        auto itr = m_map.find(key);
        if (itr == m_map.end()) {
            return m_vec.end();
        }
        return m_vec.begin() + itr->second;
    }

    void clear() {
        m_vec.clear();
        m_map.clear();
    }

private:
    std::unordered_map<Key, size_t> m_map;
    std::vector<value_type> m_vec;
};

}   // namespace intel_cpu
}   // namespace ov