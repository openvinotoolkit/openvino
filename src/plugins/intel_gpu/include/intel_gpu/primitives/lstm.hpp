// Copyright (C) 2018-2024 Intel Corporation
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

struct lstm_elt : public RNNParams<lstm_elt> {
    CLDNN_DECLARE_PRIMITIVE(lstm_elt)
    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    using RNNParams::RNNParams;
    lstm_elt() : RNNParams() {}
    lstm_elt(const lstm_elt&) = default;
    bool input_forget;
};
} // namespace cldnn
