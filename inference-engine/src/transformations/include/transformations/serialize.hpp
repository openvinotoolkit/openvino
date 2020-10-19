// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API Serialize;

}  // namespace pass
}  // namespace ngraph

// ! [function_pass:serialize_hpp]
// serialize.hpp
class ngraph::pass::Serialize : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    Serialize(const std::string& xmlPath, const std::string& binPath)
        : m_xmlPath{xmlPath}, m_binPath{binPath} {}

private:
    const std::string m_xmlPath;
    const std::string m_binPath;
};
// ! [function_pass:serialize_hpp]
