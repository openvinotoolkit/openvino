// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <memory>

#include "openvino/openvino.hpp"

#include "cache/meta/meta_info.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ICache {
public:
    virtual void update_cache(const std::shared_ptr<ov::Model>& model,
                              const std::string& source_model, bool extract_body = true) {};
    virtual void serialize_cache() {};
    virtual void reset_cache() {};

    void set_serialization_dir(const std::string& serialization_dir) {
        m_serialization_dir = serialization_dir;
    }

protected:
    size_t m_serialization_timeout = 60;
    std::string m_serialization_dir = ".";

    ICache() = default;

    bool serialize_model(const std::pair<std::shared_ptr<ov::Model>, MetaInfo>& graph_info,
                         const std::string& rel_serialization_path);
};
}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov