// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class GenerateMappingFile;

}  // namespace pass
}  // namespace ngraph

/**
 * @brief Generate mapping file based on output tensor names.
 */
class ngraph::pass::GenerateMappingFile : public ngraph::pass::FunctionPass {
    std::string m_path_to_file;
    bool m_extract_name;

public:
    OPENVINO_RTTI("GenerateMappingFile", "0");
    explicit GenerateMappingFile(const std::string& path, bool extract_name = true)
        : m_path_to_file(path),
          m_extract_name(extract_name) {}

    bool run_on_model(const std::shared_ptr<ngraph::Function>&) override;
};
