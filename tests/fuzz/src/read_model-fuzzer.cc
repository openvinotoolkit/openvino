// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"

#define COUNT_OF(A) (sizeof(A) / sizeof(A[0]))
const char kSplitSequence[] = {'F', 'U', 'Z', 'Z', '_', 'N', 'E', 'X', 'T', '_', 'F', 'I', 'E', 'L', 'D'};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    size_t split_counter = 0;
    size_t split[1] = {0};
    if (size < sizeof(kSplitSequence)) return 0;  // we at least expect one separator
    for (size_t i = 0; i < size - sizeof(kSplitSequence); i++)
        if (0 == memcmp(data + i, kSplitSequence, sizeof(kSplitSequence))) {
            split[split_counter++] = i;
            if (COUNT_OF(split) <= split_counter) break;
        }
    if (COUNT_OF(split) != split_counter) return 0;  // not enough splits

    // isolate xml data
    size_t net_size = split[0];
    std::string net((const char*)data, net_size);
    size -= net_size + sizeof(kSplitSequence);
    data += net_size + sizeof(kSplitSequence);

    // isolate weights data
    ov::Tensor weights(ov::element::u8, {static_cast<size_t>(size)}, (void*)data);

    // read xml and set weights
    try {
        ov::Core core;
        auto model = core.read_model(net, weights);
    } catch (const std::exception&) {
        return 0;  // fail gracefully on expected exceptions
    }

    return 0;
}
