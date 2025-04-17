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

struct gru_seq : public RNNParams<gru_seq> {
    CLDNN_DECLARE_PRIMITIVE(gru_seq)
    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    using RNNParams::RNNParams;
    gru_seq() : RNNParams() {
        weights = W.pid;
        input = x.pid;
    }
    gru_seq(const gru_seq&) = default;
    primitive_id input;
    primitive_id weights;
    bool linear_before_reset;
};

}  // namespace cldnn
