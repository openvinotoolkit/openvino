// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_sinking_test_utils.hpp"

#include "openvino/opsets/opset12.hpp"

using namespace ov;
using namespace ov::opset12;

void ShiftLeft(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.begin(), vec.begin() + k, buffer.begin());

    for (int i = k; i < vec.size(); ++i) {
        vec[i - k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.end() - k);
}

void ShiftRight(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.end() - k, vec.end(), buffer.begin());

    for (int i = vec.size() - 1 - k; i >= 0; --i) {
        vec[i + k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.begin());
}

std::vector<size_t> GatherForward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);
    ShiftLeft(vec, 2);
    return vec;
}

std::vector<size_t> GatherBackward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);  // Not the same as in binary tests
    ShiftRight(vec, 2);
    return vec;
}
