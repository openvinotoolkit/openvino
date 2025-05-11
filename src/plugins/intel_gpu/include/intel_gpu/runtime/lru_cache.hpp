// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <thread>
#include <mutex>

#include "kernel.hpp"

namespace cldnn {

struct primitive_impl;

/// @brief LRU cache which remove the least recently used data when cache is full.
template<typename Key, typename Value, typename KeyHasher = std::hash<Key>>
class LruCache {
public:
    using data_type = std::pair<Key, Value>;


public:
    explicit LruCache(size_t caps) : _capacity(caps) {}

    ~LruCache() {
        clear();
    }

    /**
     * @brief Get the least recently used element with key and value pair in the cache
     *
     * @return std::pair<Key, Value>
     */
    std::pair<Key, Value> get_lru_element() const {
        if (_lru_data_list.size()) {
            return _lru_data_list.back();
        } else {
            return std::make_pair(Key(), Value());
        }
    }

    /**
     * @brief Add new value with associated key into the LRU cache
     *
     * @param key if same key is existed in the cache, the value of key is updated new entry.
     * @param value
     * @return true, if cache is full and least recently used entry are removed to add new entry.
     * @return false Otherwise
     */
    bool add(const Key& key, const Value& value) {
        auto map_iter = _key_map.find(key);
        if (map_iter != _key_map.end()) {
            touch_data(map_iter->second);
            map_iter->second->second = value;
            return false;
        }

        bool popped_last_element = false;
        if (_capacity > 0 && _capacity == _key_map.size()) {
            pop();
            popped_last_element = true;
        }
        auto iter = _lru_data_list.insert(_lru_data_list.begin(), {key, value});
        _key_map.insert({key, iter});
        return popped_last_element;
    }

    /**
     * @brief Check whether the value assocaited with key is existed in the cache
     *
     * @param key
     * @return true if any value associated with the key is existed.
     * @return false otherwise
     */
    bool has(const Key& key) const {
        return (_key_map.find(key) != _key_map.end());
    }

    /**
     * @brief Find a value associated with a key
     *
     * @param key
     * @return Value a value associated with input key. if the key is not existed in the cache, return nullptr
     */
    Value get(const Key& key) {
        auto iter = _key_map.find(key);
        if (iter == _key_map.end()) {
            return Value();
        }
        touch_data(iter->second);
        return _lru_data_list.front().second;
    }

    /**
     * @brief Remove all entries
     *
     */
    void clear() {
        _lru_data_list.clear();
        _key_map.clear();
    }

    /**
     * @brief Return current size of cache
     *
     * @return size_t
     */
    size_t size() const {
        return _lru_data_list.size();
    }

    /**
     * @brief Return capacity of the cache
     *
     * @return size_t
     */
    size_t capacity() const {
        return _capacity;
    }

    /**
     * @brief Return whether the cache is full or not
     *
     * @return true, if cache is full, false otherwise
     */
    size_t is_full() const {
        return _lru_data_list.size() == _capacity;
    }

    /**
     * @brief Get the all keys object
     *
     * @return std::vector<Key>
     */
    std::vector<Key> get_all_keys() const {
        std::vector<Key> key_list;
        for (auto& iter : _lru_data_list) {
            key_list.push_back(iter.first);
        }
        return key_list;
    }

private:
    using lru_data_list_type = std::list<data_type>;
    using lru_data_list_iter = typename lru_data_list_type::iterator;

    std::list<data_type> _lru_data_list;
    std::unordered_map<Key, lru_data_list_iter, KeyHasher> _key_map;
    const size_t _capacity;

    /**
     * @brief Move data to front of list because the data is touched.
     *
     * @param iter iterator of current touched data
     */
    void touch_data(lru_data_list_iter iter) {
        _lru_data_list.splice(_lru_data_list.begin(), _lru_data_list, iter);
    }

    /**
     * @brief Pop n least recently used cache data.
     *
     * @param n number of data to be popped
     */
    void pop(size_t n = 1) {
        for (size_t i = 0; i < n && !_lru_data_list.empty(); ++i) {
            _key_map.erase(_lru_data_list.back().first);
            _lru_data_list.pop_back();
        }
    }
};

using KernelsCache = cldnn::LruCache<size_t, cldnn::kernel::ptr>;

template<typename Key, typename Value, typename KeyHasher = std::hash<Key>>
class LruCacheThreadSafe : public LruCache<Key, Value, KeyHasher> {
public:
    using parent = LruCache<Key, Value, KeyHasher>;
    using ItemType = std::pair<Key, Value>;
    using FuncRemoveItem = std::function<void(ItemType&)>;
    using parent::parent;

    explicit LruCacheThreadSafe(size_t caps) : parent(caps) { }

    bool add(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto popped_item = parent::get_lru_element();
        auto ret = parent::add(key, value);
        if (ret && _remove_popped_item) {
            _remove_popped_item(popped_item);
        }
        return ret;
    }

    bool has(const Key& key) const {
        std::lock_guard<std::mutex> lock(_mutex);
        return parent::has(key);
    }

    Value get(const Key& key) {
        std::lock_guard<std::mutex> lock(_mutex);
        return parent::get(key);
    }

    void set_remove_item_callback(FuncRemoveItem callback) {
        _remove_popped_item = std::move(callback);
    }

private:
    FuncRemoveItem _remove_popped_item;
    mutable std::mutex _mutex;
};

}  // namespace cldnn
