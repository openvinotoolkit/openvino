// Copyright (C) 2021 Intel Corporation
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

class ngraph::pass::GenerateMappingFile: public ngraph::pass::FunctionPass {
    std::string m_path_to_file;
public:
    NGRAPH_RTTI_DECLARATION;
    explicit GenerateMappingFile(const std::string & path) : m_path_to_file(path) {}

    bool run_on_function(std::shared_ptr<ngraph::Function>) override;
};
