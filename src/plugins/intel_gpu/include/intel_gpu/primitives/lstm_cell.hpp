// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "activation.hpp"
#include <vector>
#include <algorithm>
#include "intel_gpu/graph/serialization/activation_serializer.hpp"
#include "rnn.hpp"


namespace cldnn {

struct lstm_cell : public RNNParams<lstm_cell> {
    CLDNN_DECLARE_PRIMITIVE(lstm_cell)
    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    using RNNParams::RNNParams;
    lstm_cell(const lstm_cell&) = default;
    lstm_cell() : RNNParams() {}
};
}  // namespace cldnn
