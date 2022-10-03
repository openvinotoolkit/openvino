// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "executable.hpp"
#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
class LRUCache : public std::enable_shared_from_this<LRUCache> {
public:
    using GraphCache = std::unordered_map<std::string, std::shared_ptr<Executable>>;
    using ClonedFunctionMap = std::unordered_map<std::string, std::shared_ptr<Function>>;

    LRUCache();

    virtual ~LRUCache();

    void add_entry(const std::vector<int>& shape, std::shared_ptr<Executable> exec, std::shared_ptr<Function> func);
    bool is_cached(const std::vector<int>& shape);
    std::shared_ptr<Executable> get_cached_entry(const std::vector<int>& shape);
    void convert_shape_to_string(const std::vector<int>& shape, std::ostringstream& key);
    std::shared_ptr<Function> get_cloned_function(const std::vector<int>& shape);

private:
    int m_cache_size;
    GraphCache m_map;
    ClonedFunctionMap m_clone_function_map;
    std::list<std::vector<int>> m_list;
    std::mutex m_mutex;
};
}  // namespace runtime
}  // namespace ngraph
