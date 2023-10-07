// Copyright (C) 2018-2023 Intel Corporation
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

    virtual std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                                bool is_extract_body = true,
                                                bool is_copy_constants = true) {
        return std::list<ExtractedPattern>{};
    }

    void set_extractor_name(const std::string& _extractor_name) { extractor_name = _extractor_name; }
    void set_match_coefficient(float _match_coefficient) {
        if (_match_coefficient  < 0 || _match_coefficient > 1) {
            throw std::runtime_error("[ ERROR ] Match coefficient should be from 0 to 1!");
        }
        match_coefficient = _match_coefficient; 
    }

protected:
    std::string extractor_name = "";
    float match_coefficient = 0.9f;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
