// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "utils/model_comparator.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class SubgraphExtractor {
public:
    using Ptr = std::shared_ptr<SubgraphExtractor>;
    // ov_model, input_info, extractor_name
    using ExtractedPattern = std::tuple<std::shared_ptr<ov::Model>, std::map<std::string, ov::conformance::InputInfo>, std::string>;

    virtual std::vector<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) {
        return std::vector<ExtractedPattern>{};
    }

    void set_extractor_name(const std::string& _extractor_name) { extractor_name = _extractor_name; }
    void set_extract_body(bool _is_extract_body) { is_extract_body = _is_extract_body; }
    void set_save_const(bool _is_save_const) { is_save_const = _is_save_const; }

protected:
    std::string extractor_name = "";
    bool is_extract_body = true, is_save_const = true;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
