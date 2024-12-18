// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace intel_cpu {

class JitKernel;

enum class WeightCompressionType {
    FP16 = 0,
    INT8,
    INT4
};
class ActSparseFcKernel {
public:
    // compile time parameters
    ActSparseFcKernel(WeightCompressionType wtype, bool with_zero_points, int ic_group_size);

    void operator()(const float* input,
                    float* output,
                    int M,
                    int IC,
                    int OC,
                    float threshold,
                    float zero_point,
                    const void* W,
                    const float* scales,
                    const uint8_t* zp);

    void repack_weights_i4(uint8_t * src, uint8_t * dst, int IC, int OC);

private:
    const WeightCompressionType m_wtype;
    const bool m_with_zp;
    JitKernel* m_decompzp_kernel;
    JitKernel* m_accumulate_kernel;
    const int m_ic_group_size;

    std::vector<int> m_nonzero_ids;
    std::vector<float> m_nonzero_val;
    std::vector<float> m_output_temp;
    int m_nonzero_cnt;
};

}  // namespace intel_cpu
}  // namespace ov
