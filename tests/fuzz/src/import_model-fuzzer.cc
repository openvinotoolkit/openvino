// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"

#include <stdio.h>
#include <fstream>
#include <string>

#define COUNT_OF(A) (sizeof(A) / sizeof(A[0]))
const char kSplitSequence[] = {'F', 'U', 'Z', 'Z', '_', 'N', 'E', 'X', 'T', '_', 'F', 'I', 'E', 'L', 'D'};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    using namespace std;

    size_t kSplitSequenceSize = sizeof(kSplitSequence);
    size_t split_counter = 0;
    size_t split[1] = {0};
    if (size < kSplitSequenceSize) return 0;  // we at least expect one separator
    for (size_t i = 0; i < size - kSplitSequenceSize && split_counter < COUNT_OF(split); i++)
        if (0 == memcmp(data + i, kSplitSequence, kSplitSequenceSize)) {
            split[split_counter++] = i;
        }
    if (COUNT_OF(split) != split_counter) return 0;  // not enough splits

    if(split[0] < sizeof(ov::pass::StreamSerialize::DataHeader))
         return 0;

    unsigned int _size = static_cast<unsigned int>(split[0]);
    const uint8_t* dev_name_data = data + (_size + kSplitSequenceSize);
    string device_name(reinterpret_cast<const char*>(dev_name_data), size - (_size + kSplitSequenceSize));

    struct ov::pass::StreamSerialize::DataHeader* dh = (struct ov::pass::StreamSerialize::DataHeader*)data;
    size_t total = dh->custom_data_size + dh->consts_size + dh->model_size;

    if(total != _size || dh->custom_data_offset + dh->custom_data_size > total || dh->consts_offset + dh->consts_size > total || dh->model_offset + dh->model_size > total)
        return 0;

    try {
        stringstream ss;
        ss.write(reinterpret_cast<char const*>(data), _size);
        iostream& io = ss;

        ov::Core core;
        ov::AnyMap properties;
        core.import_model(io, device_name.c_str(), properties);
    } catch (const exception&) {
        return 0;  // fail gracefully on expected exceptions
    }

    return 0;
}
