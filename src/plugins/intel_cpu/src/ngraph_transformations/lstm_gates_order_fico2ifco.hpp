// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class LSTMGatesOrderFICO2IFCO : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("LSTMGatesOrderFICO2IFCO", "0");
    LSTMGatesOrderFICO2IFCO();
};

}   // namespace intel_cpu
}   // namespace ov
