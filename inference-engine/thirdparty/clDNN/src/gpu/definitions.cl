/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#ifdef CODE_PREFIX
#define CODE_BEGIN CODE_PREFIX
#define CODE_END CODE_POSTFIX
#else
#define CODE_BEGIN
#define CODE_END
#endif

CODE_BEGIN
enum neural_memory_format {
    x_f32,
    xb_f32,     // 1D+batch, float32
    bx_f32,     // 1D+batch, float32
    yxfb_f32,   // 3D+batch, float32
    byxf_f32,   // for convolution_cpu_jit_batch1
    bfyx_f32,   // used in Caffe
    fyxb_f32,   // used in Caffe
    oiyx_f32,   // format used only for weights: o - output feature maps, i - input feature maps
    byxf_b24_f32,        // for convolution_cpu_generic
    yxoi_o4_f32,       // for convolution_cpu_generic
    os_yxi_sv16_f32,   // format used only for weights: os - output slice, i - input feature maps, sv16 - 16 values of single slice
    bs_yxf_bv24_f32,
    any=-1
};

#pragma pack(push, 4)
typedef struct _neural_memory_tag {
    uint format;
    uint feature_offset;
    uint spatial_offset;
    uint vector_size;
    uint data_offset;
    uint data[1];
} neural_memory;

typedef struct _neural_vector_tag {
    uint feature_offset;
    uint spatial_offset;
    uint raw_size;
    uint data[1];
} neural_vector;
#pragma pack(pop)

// neural_memory accessors
__attribute__((overloadable)) __global uint* get_raw(__global neural_memory* mem) { return &(mem->data[0]); }
__attribute__((overloadable)) const __global uint* get_raw(const __global neural_memory* mem) { return &(mem->data[0]); }
__attribute__((overloadable)) uint get_raw_size(const __global neural_memory* mem) { return mem->vector_size; } 

__attribute__((overloadable)) __global uint* get_batch(__global neural_memory* mem) { return get_raw(mem); }
__attribute__((overloadable)) const __global uint* get_batch(const __global neural_memory* mem) { return get_raw(mem); }
__attribute__((overloadable)) uint get_batch_size(const __global neural_memory* mem) { return mem->feature_offset; }

__attribute__((overloadable)) __global uint* get_feature(__global neural_memory* mem) { return &(mem->data[mem->feature_offset]); }
__attribute__((overloadable)) const __global uint* get_feature(const __global neural_memory* mem) { return &(mem->data[mem->feature_offset]); }
__attribute__((overloadable)) uint get_feature_size(const __global neural_memory* mem) { return mem->spatial_offset - mem->feature_offset; }

__attribute__((overloadable)) __global uint* get_spatial(__global neural_memory* mem) { return &(mem->data[mem->spatial_offset]); }
__attribute__((overloadable)) const __global uint* get_spatial(const __global neural_memory* mem) { return &(mem->data[mem->spatial_offset]); }
__attribute__((overloadable)) uint get_spatial_size(const __global neural_memory* mem) { return get_raw_size(mem) - mem->spatial_offset; } 

__attribute__((overloadable)) __global void* get_data(__global neural_memory* mem) { return &(mem->data[mem->data_offset]); }
__attribute__((overloadable)) const __global void* get_data(const __global neural_memory* mem) { return &(mem->data[mem->data_offset]); }
__attribute__((overloadable)) size_t get_element_size(const __global neural_memory* mem) { return sizeof(float); }

__attribute__((overloadable)) size_t get_data_size(const __global neural_memory* mem) {
    size_t result = get_element_size(mem);

    const __global uint* raw = get_raw(mem);
    uint raw_size = get_raw_size(mem);

    for(uint i = 0; i < raw_size; i++) {
        result *= raw[i];
    }
    return result;
}

// neural_vector accessors
// TODO NOTE: non-const accessors are disabled now, because read-only neural_vector argument is only supported now

//__attribute__((overloadable)) __global uint* get_raw(__global neural_vector* v) { return &(v->data[0]); }
__attribute__((overloadable)) const __global uint* get_raw(const __global neural_vector* v) { return &(v->data[0]); }
__attribute__((overloadable)) uint get_raw_size(const __global neural_vector* v) { return v->raw_size; } 

//__attribute__((overloadable)) __global uint* get_batch(__global neural_vector* v) { return get_raw(v); }
__attribute__((overloadable)) const __global uint* get_batch(const __global neural_vector* v) { return get_raw(v); }
__attribute__((overloadable)) uint get_batch_size(const __global neural_vector* v) { return v->feature_offset; }

//__attribute__((overloadable)) __global uint* get_feature(__global neural_vector* v) { return &(v->data[v->feature_offset]); }
__attribute__((overloadable)) const __global uint* get_feature(const __global neural_vector* v) { return &(v->data[v->feature_offset]); }
__attribute__((overloadable)) uint get_feature_size(const __global neural_vector* v) { return v->spatial_offset - v->feature_offset; }

//__attribute__((overloadable)) __global uint* get_spatial(__global neural_vector* v) { return &(v->data[v->spatial_offset]); }
__attribute__((overloadable)) const __global uint* get_spatial(const __global neural_vector* v) { return &(v->data[v->spatial_offset]); }
__attribute__((overloadable)) uint get_spatial_size(const __global neural_vector* v) { return get_raw_size(v) - v->spatial_offset; } 

CODE_END

/*
KERNEL(Fully_Connected_GPU)
DECALRE_CONSTANT()
BEGIN_ARGUMENTS_DECLARATION
DECLARE_INPUT_MEMORY_ARGUMENT(input_mem)
DECLARE_INPUT_MEMORY_ARGUMENT(weights_mem)
DECLARE_INPUT_MEMORY_ARGUMENT(bias_mem)
DECLARE_OUTPUT_MEMORY_ARGUMENT(dst_mem)
END_ARGUMENTS_DECLARATION
CODE_BEGIN
#define WEIGHTS { 1.0, 3.2, 4.5, 6.7 }
#define WEIGHTS_SIZE { 2, 2 }
#define WEIGHTS_DIM 2
*/
__kernel void Fully_Connected_GPU(__global neural_memory* input_mem, __global neural_memory* weights_mem, __global neural_memory* bias_mem, __global neural_memory* dst_mem)
{
    __global uint* input_size = get_raw(input_mem);
    __global uint* weights_size = get_raw(weights_mem);
    __global float* input = (__global float*)get_data(input_mem);
    __global float* weights = (__global float*)get_data(weights_mem);
    __global float* bias = (__global float*)get_data(bias_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int x = get_global_id(0);

    pDst[x] = 0;
    uint outXIdx = x / input_size[0];
    uint inputBatchIdx = x % input_size[0];
    uint weightYIdx = outXIdx * weights_size[0];
    for (uint i = 0; i < input_size[2]; i++)
    {
        pDst[x] += input[i * input_size[0] + inputBatchIdx] * weights[weightYIdx + i];
    }
    pDst[x] += bias[outXIdx];
}
CODE_END

CODE_BEGIN
__kernel void Convolution_GPU(
    const __global neural_memory* input_mem,
    const __global neural_memory* filter_mem,
    float bias,
    __global neural_memory* dst_mem,
    const __global neural_vector* spatial_stride)
{

//
    const __global uint* input_size = get_raw(input_mem);
    const __global uint* filter_size = get_raw(filter_mem);
    const __global uint* dst_size = get_raw(dst_mem);
    const __global float* input = (const __global float*)get_data(input_mem);
    const __global float* filter = (const __global float*)get_data(filter_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);
//

    int global_id = get_global_id(0);
    const int batch_num = dst_size[0];
    const int batch_offset = global_id % dst_size[0];

    const int idx = global_id / batch_num;

    const int x = (idx % input_size[2]) * get_spatial(spatial_stride)[0];
    const int y = (idx * get_spatial(spatial_stride)[1]) / input_size[2];

    const int out_offset = idx * batch_num + batch_offset;

    pDst[out_offset] = 0;
    for (uint i = 0; i < filter_size[4]; i++)
    {
        for (uint j = 0; j < filter_size[3]; j++)
        {
            int input_idx = (x + j + ((y + i) * input_size[2])) * batch_num + batch_offset;
            int filter_idx = i * filter_size[3] + j;
            pDst[out_offset] += input[input_idx] * filter[filter_idx];
        }
    }
    pDst[out_offset] += bias;
}
CODE_END
