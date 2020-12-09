// TODO: c++17 code
// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <utility>
#include <vector>
#include <string>
//#include <filesystem>
#include <any>
#include <memory>
#include <ngraph/ngraph.hpp>

namespace SubgraphsDumper {

class OPCache {
public:
    OPCache() : num_neighbours_to_cache(0) {
        m_ops_cache = std::vector<std::pair<std::shared_ptr<ngraph::Node>, size_t>>();
    }

    static std::unique_ptr<OPCache> make_cache() {
        return std::unique_ptr<OPCache>(new OPCache());
    }

    void update_ops_cache(const std::shared_ptr<ngraph::Node> &op, const std::string &source_model = "");
    void update_ops_cache(const std::shared_ptr<ngraph::Function> &func, const std::string &source_model = "");
    void serialize_cached_ops(const std::string &serialization_dir);
    void set_num_neighbours_to_cache(size_t num) { num_neighbours_to_cache = num; }

protected:
    std::vector<std::pair<std::shared_ptr<ngraph::Node>, size_t>> m_ops_cache;

    size_t num_neighbours_to_cache = 0;
};
}  // namespace SubgraphsDumper