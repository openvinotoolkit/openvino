// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct argmax_conf {
    bool out_max_val_;
    int top_k_;
    bool has_axis_;
    int axis_index_;
};

namespace XARCH {

void arg_max_execute(const float* inputs, float *outputs, std::vector<size_t> dims, argmax_conf& conf);

}  // namespace XARCH

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
