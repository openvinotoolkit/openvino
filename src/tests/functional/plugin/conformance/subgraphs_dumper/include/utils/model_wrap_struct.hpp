// Copyright (C) 2019-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/ov_plugin_cache.hpp"

namespace SubgraphsDumper {

struct Model {
    std::string path;
    size_t size = 0;

    Model(std::string model) {
        path = model;
        try {
            size = ov::test::utils::PluginCache::get().core()->read_model(path)->get_graph_size();
        } catch (...) {
            std::cout << "Impossible to read network: " << path << std::endl;
        }
    }

    bool operator<(const Model &m) const {
        return size < m.size;
    }

    bool operator>(const Model &m) const {
        return size > m.size;
    }
};
}  // namespace SubgraphsDumper