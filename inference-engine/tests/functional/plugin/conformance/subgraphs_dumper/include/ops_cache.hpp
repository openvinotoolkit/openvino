// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "matchers/matchers_manager.hpp"
#include "functional_test_utils/include/functional_test_utils/layer_test_utils/op_info.hpp"

namespace SubgraphsDumper {

class OPCache {
public:
    OPCache() : num_neighbours_to_cache(0), manager(MatchersManager()),
                m_ops_cache(std::vector<std::pair<std::shared_ptr<ngraph::Node>, LayerTestsUtils::OPInfo>>()) {}

    static std::unique_ptr<OPCache> make_cache() {
        return std::unique_ptr<OPCache>(new OPCache());
    }

    void update_ops_cache(const std::shared_ptr<ngraph::Node> &op, const std::string &source_model = {});

    void update_ops_cache(const std::shared_ptr<ngraph::Function> &func, const std::string &source_model = {});

    void serialize_cached_ops(const std::string &serialization_dir);

    void set_num_neighbours_to_cache(size_t num) { num_neighbours_to_cache = num; }

    void serialize_meta_info(const LayerTestsUtils::OPInfo &info, const std::string &path);

    float get_size_of_cached_ops();

protected:
    std::vector<std::pair<std::shared_ptr<ngraph::Node>, LayerTestsUtils::OPInfo>> m_ops_cache;
    MatchersManager manager;
    size_t num_neighbours_to_cache = 0;
    enum SerializationStatus {
        OK = 0,
        FAILED = 1,
        RETRY = 2,
    };
    SerializationStatus serialize_function(const std::pair<std::shared_ptr<ngraph::Node>, LayerTestsUtils::OPInfo> &op_info,
                            const std::string &serialization_dir);
};
}  // namespace SubgraphsDumper
