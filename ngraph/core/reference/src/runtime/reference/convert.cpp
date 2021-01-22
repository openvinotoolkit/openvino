//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/runtime/reference/convert.hpp"
#include "jit_generator.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                class jit_convert_u8_to_f16 : public jit::Generator
                {
                    jit_convert_u8_to_f16()
                    {
                        using namespace Xbyak;

                        auto u8vec = xmm1;
                        auto i32vec = ymm2;
                        auto f16vec = xmm3;
                        auto fvec = ymm4;

                        auto reg_src = rax;
                        auto reg_dst = rbx;
                        auto reg_sz = rdx;

                        Label loop;

                        preamble();

                        mov(reg_src, ptr[param + offsetof(args_t, arg)]);
                        mov(reg_dst, ptr[param + offsetof(args_t, out)]);
                        mov(reg_sz, ptr[param + offsetof(args_t, count)]);

                        L(loop);

                        movq(u8vec, qword[reg_src]);
                        vpmovzxbd(i32vec, u8vec);
                        vcvtdq2ps(fvec, i32vec);
                        vcvtps2ph(f16vec, fvec, 0);
                        movdqu(xword[reg_dst], f16vec);

                        lea(reg_src, ptr[reg_src + sizeof(uint8_t) * 8]);
                        lea(reg_dst, ptr[reg_dst + sizeof(float16) * 8]);

                        dec(reg_sz);
                        test(reg_sz, reg_sz);
                        jnz(loop);

                        postamble();
                    }

                public:
                    typedef struct
                    {
                        const uint8_t* arg;
                        float16* out;
                        const size_t count;
                    } args_t;

                    typedef void (*fn_t)(const args_t*);

                    static fn_t get()
                    {
                        if (mayiuse(avx) && mayiuse(avx2) && mayiuse(fp16))
                        {
                            static jit_convert_u8_to_f16 generator;
                            return (fn_t)generator.getCode();
                        }
                        return nullptr;
                    }
                };

                class jit_convert_f16_to_f32 : public jit::Generator
                {
                    jit_convert_f16_to_f32()
                    {
                        using namespace Xbyak;

                        auto f16vec = xmm1;
                        auto f32vec = ymm2;

                        auto reg_src = rax;
                        auto reg_dst = rbx;
                        auto reg_sz = rdx;

                        Label loop;

                        preamble();

                        mov(reg_src, ptr[param + offsetof(args_t, arg)]);
                        mov(reg_dst, ptr[param + offsetof(args_t, out)]);
                        mov(reg_sz, ptr[param + offsetof(args_t, count)]);

                        L(loop);

                        movdqu(f16vec, xword[reg_src]);
                        vcvtph2ps(f32vec, f16vec);
                        vmovups(yword[reg_dst], f32vec);

                        lea(reg_src, ptr[reg_src + sizeof(float16) * 8]);
                        lea(reg_dst, ptr[reg_dst + sizeof(float) * 8]);

                        dec(reg_sz);
                        test(reg_sz, reg_sz);
                        jnz(loop);

                        postamble();
                    }

                public:
                    typedef struct
                    {
                        const float16* arg;
                        float* out;
                        const size_t count;
                    } args_t;

                    typedef void (*fn_t)(const args_t*);

                    static fn_t get()
                    {
                        if (mayiuse(avx) && mayiuse(fp16))
                        {
                            static jit_convert_f16_to_f32 generator;
                            return (fn_t)generator.getCode();
                        }
                        return nullptr;
                    }
                };
            } // namespace

            template <>
            void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count)
            {
                size_t i = 0;
                const size_t step = 8;
                const size_t n = count / step;

                if (n)
                {
                    auto converter = jit_convert_u8_to_f16::get();

                    if (converter)
                    {
                        jit_convert_u8_to_f16::args_t args = {arg, out, n};
                        converter(&args);
                        i = n * step;
                    }
                }

                for (; i < count; ++i)
                {
                    out[i] = static_cast<float16>(arg[i]);
                }
            }

            template <>
            void convert<float16, float>(const float16* arg, float* out, size_t count)
            {
                size_t i = 0;
                const size_t step = 8;
                const size_t n = count / step;

                if (n)
                {
                    auto converter = jit_convert_f16_to_f32::get();

                    if (converter)
                    {
                        jit_convert_f16_to_f32::args_t args = {arg, out, n};
                        converter(&args);
                        i = n * step;
                    }
                }

                for (; i < count; ++i)
                {
                    out[i] = static_cast<float>(arg[i]);
                }
            }
        }
    }
}
