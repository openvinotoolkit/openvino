// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertCellsToCellsIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertCellsToCellsIE: public ngraph::pass::GraphRewrite {
public:
    ConvertCellsToCellsIE() : GraphRewrite() {
        convert_lstm_cell();
        convert_gru_cell();
        convert_rnn_cell();
    }

private:
    void convert_lstm_cell();
    void convert_gru_cell();
    void convert_rnn_cell();
};
