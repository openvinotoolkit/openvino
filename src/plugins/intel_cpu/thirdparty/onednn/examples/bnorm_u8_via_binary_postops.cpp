/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @example bnorm_u8_via_binary_postops.cpp
/// @copybrief bnorm_u8_via_binary_postops_cpp
/// > Annotated version: @ref bnorm_u8_via_binary_postops_cpp
///
/// @page bnorm_u8_via_binary_postops_cpp_short
/// Bnorm u8 via binary postops example.
///
/// @page bnorm_u8_via_binary_postops_cpp Bnorm u8 by binary post-ops example
/// The example implements the Batch normalization u8 via the following
/// operations: binary_sub(src, mean), binary_div(tmp_dst, variance),
/// binary_mul(tmp_dst, scale), binary_add(tmp_dst, shift).
///
/// Some key take-aways include:
///
/// * How tensors are implemented and submitted to primitives.
/// * How primitives are created.
/// * How to use multiple binary post operations.
/// * How to use different data types in binary.
///
/// @include bnorm_u8_via_binary_postops.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void bnorm_u8_via_binary_postops(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 150, // tensor height
            IW = 150; // tensor width

    // Tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims params_dims = {1, IC, 1, 1};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> mean_data(product(params_dims));
    std::vector<float> variance_data(product(params_dims));
    std::vector<float> scale_data(product(params_dims));
    std::vector<float> shift_data(product(params_dims));
    std::vector<float> oscale_data(product(params_dims));

    // Initialize
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(mean_data.begin(), mean_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(variance_data.begin(), variance_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 4.f);
    });
    std::generate(scale_data.begin(), scale_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 6.f);
    });
    std::generate(shift_data.begin(), shift_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 8.f);
    });
    std::generate(oscale_data.begin(), oscale_data.end(), []() { return 0.5; });

    // Create descriptors.
    auto src_md = memory::desc(src_dims, dt::u8, tag::nhwc);
    auto mean_md = memory::desc(params_dims, dt::f32, tag::nhwc);
    auto variance_md = memory::desc(params_dims, dt::f32, tag::nhwc);
    auto scale_md = memory::desc(params_dims, dt::f32, tag::nhwc);
    auto shift_md = memory::desc(params_dims, dt::f32, tag::nhwc);
    auto oscale_md = memory::desc(params_dims, dt::f32, tag::nhwc);

    // Create src memory objects.
    auto src_mem = memory(src_md, engine);
    auto mean_mem = memory(mean_md, engine);
    auto variance_mem = memory(variance_md, engine);
    auto scale_mem = memory(scale_md, engine);
    auto shift_mem = memory(shift_md, engine);
    auto oscale_mem = memory(oscale_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(mean_data.data(), mean_mem);
    write_to_dnnl_memory(variance_data.data(), variance_mem);
    write_to_dnnl_memory(scale_data.data(), scale_mem);
    write_to_dnnl_memory(shift_data.data(), shift_mem);
    write_to_dnnl_memory(oscale_data.data(), oscale_mem);

    // Create operation descriptor.
    // dst_tmp = src - mean
    auto binary_d
            = binary::desc(algorithm::binary_sub, src_md, mean_md, src_md);

    // Bnorm operation with scale and shift
    post_ops binary_ops;
    // dst_tmp = dst_tmp / variance
    binary_ops.append_binary(algorithm::binary_div, variance_md);
    // dst_tmp = dst_tmp * scale
    binary_ops.append_binary(algorithm::binary_mul, scale_md);
    // dst_tmp = dst_tmp + shift
    binary_ops.append_binary(algorithm::binary_add, shift_md);
    // dst = dst_tmp * output_scale (only for re-quantization)
    binary_ops.append_binary(algorithm::binary_mul, oscale_md);
    primitive_attr binary_attr;
    binary_attr.set_post_ops(binary_ops);

    // Create primitive descriptor.
    auto binary_pd = binary::primitive_desc(binary_d, binary_attr, engine);

    // Create the primitive.
    auto binary_prim = binary(binary_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0, src_mem});
    binary_args.insert({DNNL_ARG_SRC_1, mean_mem});
    // In-place mode (dst is src)
    binary_args.insert({DNNL_ARG_DST, src_mem});
    binary_args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, variance_mem});
    binary_args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, scale_mem});
    binary_args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1, shift_mem});
    binary_args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1, oscale_mem});

    // Primitive execution
    binary_prim.execute(engine_stream, binary_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            bnorm_u8_via_binary_postops, parse_engine_kind(argc, argv));
}
