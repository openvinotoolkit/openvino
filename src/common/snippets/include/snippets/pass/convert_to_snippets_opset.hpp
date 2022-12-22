// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include <snippets/itt.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

 /**
 * @interface ConvertToSnippetsOpset
 * @ingroup snippets
 * @brief ConvertToSnippetsOpset transformation converts operations to snippets opset.
 *
 * ConvertToSnippetsOpset transformation converts operations to snippets opset. 
 * If new operation shanges output preccisions then the transformation restores precision for each output.
 */
class ConvertToSnippetsOpset : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertToSnippetsOpset", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
