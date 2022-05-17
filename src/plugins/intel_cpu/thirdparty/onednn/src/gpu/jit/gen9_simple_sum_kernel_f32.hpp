/*******************************************************************************
 * Copyright 2019-2021 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef GPU_JIT_GEN9_SIMPLE_SUM_KERNEL_F32_HPP
#define GPU_JIT_GEN9_SIMPLE_SUM_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class gen9_simple_sum_kernel_f32_t : public jit_generator<gpu_gen9> {
public:
    gen9_simple_sum_kernel_f32_t() : jit_generator<gpu_gen9>() {
        using namespace ngen;
        constexpr auto GlobalPtr = ExternalArgumentType::GlobalPtr;
        constexpr auto Scalar = ExternalArgumentType::Scalar;

        newArgument("input", GlobalPtr);
        newArgument("output", GlobalPtr);
        newArgument("scale", DataType::f, Scalar);
        newArgument("a", DataType::d, Scalar);
        externalName("ngen_gen9_simple_sum");
        finalizeInterface();

        setDefaultNoMask();

        Label append, done;

        auto global_id0_arg = r0.ud(1);
        auto src_ptr = r32;
        auto dst_ptr = r34;
        auto global_id = r33;

        auto src = r40;
        auto dst = r42;
        auto factor = r41;
        auto sum = r43;

        mov<uint32_t>(1, global_id, global_id0_arg);
        mov<uint64_t>(1, src_ptr, getArgument("input"));
        mov<uint64_t>(1, dst_ptr, getArgument("output"));
        mov<float>(1, factor, getArgument("scale"));

        mul<uint32_t>(1, global_id, global_id, 4);
        add<uint32_t>(1, src_ptr, src_ptr, global_id);
        add<uint32_t>(1, dst_ptr, dst_ptr, global_id);

        load(1, src, scattered_dword(), A64, src_ptr);
        mul<float>(1, sum, factor, src);

        cmp(1 | eq | f0[0], null.ud(), getArgument("a"), 0);
        jmpi(1 | ~f0[0], append);
        store(1, scattered_dword(), A64, dst_ptr, sum);
        jmpi(1, done);

        mark(append);
        load(1, dst, scattered_dword(), A64, dst_ptr);
        add<float>(1, sum, sum, dst);
        store(1, scattered_dword(), A64, dst_ptr, sum);

        mark(done);
        mov<uint32_t>(8, r127, r0);
        threadend(r127);
    }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_GEN9_SIMPLE_SUM_KERNEL_F32_HPP
