// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/interpolate_config.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/primitive_attr.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/precision_support.h"

namespace ov::intel_cpu {
    static inline bool isFloatCompatible(ov::element::Type prc) {
        return any_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::f64);
    }

    class InterpolateRefExecutor : public InterpolateExecutorBase {
    public:
        InterpolateRefExecutor(const InterpolateAttrs& interpAttrs,
                               const VectorDims& srcDims,
                               const VectorDims& dstDims,
                               const std::vector<float>& _dataScales)
            : InterpolateExecutorBase(interpAttrs, srcDims, dstDims, _dataScales),
              antialias(interpAttrs.antialias),
              dataScales(_dataScales),
              refInterpAttrs(interpAttrs) {}

        void exec(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_) override;

    private:
        void
        NNRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
        void linearOnnxRef(const uint8_t* in_ptr_,
                           uint8_t* out_ptr_,
                           int B,
                           int C,
                           int ID,
                           int IH,
                           int IW,
                           int OD,
                           int OH,
                           int OW);

        void cubicRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int IH, int IW, int OH, int OW);
        void linearInterpolation(const uint8_t* in_ptr_,
                                 uint8_t* out_ptr_,
                                 int B,
                                 int C,
                                 int ID,
                                 int IH,
                                 int IW,
                                 float fx,
                                 float fy,
                                 float fz,
                                 int OD,
                                 int OH,
                                 int OW,
                                 int kernel_width,
                                 bool antialias);
        void pillowRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int IH, int IW, int OH, int OW);
        void
        pillowRefNCHWAsNHWC(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

        static float getValue(const uint8_t* base, size_t offset, ov::element::Type prec);
        static void setValue(uint8_t* base, size_t offset, float value, ov::element::Type prec);

        bool antialias;
        std::vector<float> dataScales;
        InterpolateAttrs refInterpAttrs;
    };
}
