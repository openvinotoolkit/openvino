// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml.h"

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    for (const auto type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_NVFP4}) {
        const auto traits = ggml_get_type_traits(type);
        constexpr size_t count = 4 * 256;
        const size_t blocks = count / traits->blck_size;
        std::vector<uint8_t> bytes(blocks * traits->type_size);
        for (size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = uint8_t(i * 73 + (i / 7) * 11 + 19);
        for (size_t block = 0; block < blocks; ++block) {
            if (type == GGML_TYPE_IQ2_XXS) {
                const auto scale = ggml_fp32_to_fp16(0.0125f * float(block + 1));
                std::memcpy(bytes.data() + block * traits->type_size, &scale, sizeof(scale));
            } else {
                for (size_t group = 0; group < 4; ++group)
                    bytes[block * traits->type_size + group] = uint8_t((block * 4 + group) % 128);
            }
        }
        std::vector<float> output(count);
        traits->to_float(bytes.data(), output.data(), count);
        const std::string prefix = std::string(argv[1]) + "/" + traits->type_name;
        std::ofstream(prefix + "_qbytes.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        std::ofstream(prefix + "_deq.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
}
