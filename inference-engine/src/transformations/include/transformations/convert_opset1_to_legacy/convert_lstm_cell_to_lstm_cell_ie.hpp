// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertLSTMCellToLSTMCellIE);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLSTMCellToLSTMCellIE: public ngraph::pass::GraphRewrite {
public:
    ConvertLSTMCellToLSTMCellIE() : GraphRewrite() {
        convert_lstm_cell();
    }

private:
    void convert_lstm_cell();
};
