// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache.hpp"

#include "ngraph/env_util.hpp"

using namespace ngraph;
using namespace std;

// Constructor
runtime::LRUCache::LRUCache() {
    m_cache_size = 1024;

    m_map = {};
    m_list = {};
}

// Destructor
runtime::LRUCache::~LRUCache() {
    m_list.clear();
    m_map.clear();
    m_clone_function_map.clear();
}

void runtime::LRUCache::convert_shape_to_string(const vector<int>& shape, ostringstream& key) {
    if (!shape.empty()) {
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<int>(key, ", "));
    }
}

void runtime::LRUCache::add_entry(const vector<int>& shape,
                                  shared_ptr<runtime::Executable> exec,
                                  shared_ptr<Function> func) {
    std::lock_guard<std::mutex> guard(m_mutex);
    ostringstream key;
    // check if the list is empty
    if (m_list.size() == static_cast<size_t>(m_cache_size)) {
        ostringstream key;
        convert_shape_to_string(m_list.back(), key);
        m_list.pop_back();
        m_map.erase(key.str());
    }

    convert_shape_to_string(shape, key);
    m_map.insert({key.str(), exec});
    m_list.push_front(shape);
    m_clone_function_map.insert({key.str(), func});
}

bool runtime::LRUCache::is_cached(const vector<int>& shape) {
    for (auto itr = m_list.begin(); itr != m_list.end(); itr++) {
        if (*itr == shape) {
            return true;
        }
    }
    return false;
}

shared_ptr<runtime::Executable> runtime::LRUCache::get_cached_entry(const vector<int>& shape) {
    std::lock_guard<std::mutex> guard(m_mutex);
    ostringstream key;
    convert_shape_to_string(shape, key);

    // find the entry and return the function
    auto it = m_map.find(key.str());
    if (it == m_map.end()) {
        throw ngraph_error("Entry not found in cache");
    } else {
        // update list to push this reference to the front
        for (auto itr = m_list.begin(); itr != m_list.end(); itr++) {
            if (*itr == shape) {
                m_list.remove(shape);
                m_list.push_front(shape);
                break;
            }
        }
        return it->second;
    }
}

// Need the clone function to get the output shape so that
// storage can be allocated for output
shared_ptr<Function> runtime::LRUCache::get_cloned_function(const vector<int>& shape) {
    std::lock_guard<std::mutex> guard(m_mutex);
    ostringstream key;
    convert_shape_to_string(shape, key);
    // find the entry and return the function
    auto it = m_clone_function_map.find(key.str());
    if (it == m_clone_function_map.end()) {
        throw ngraph_error("Cloned function not found");
    }
    return it->second;
}
