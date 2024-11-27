// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <array>

#include "openvino/runtime/make_tensor.hpp"

#include "util.hpp"

namespace {

#define ASSERT_NO_THROW_WITH_MESSAGE(code) do{ \
    try {\
     code;\
     }catch (const std::exception &ex ) {\
         FAIL()<<ex.what();\
     }catch (...) {\
         FAIL() << "Unknown exception";\
     }\
}while(0)

#define ASSERT_NO_THROW_IF(condition, code) do { \
if (condition) {ASSERT_NO_THROW_WITH_MESSAGE(code);} else {ASSERT_ANY_THROW(code);} \
}while(0);

namespace details {

inline int8_t hi4(int8_t x) {
    return ((x & (1 << 7)) >> 4) | ((x & (1 << 6)) >> 4) | ((x & (1 << 5)) >> 4) | ((x & (1 << 4)) >> 4);
}

inline int8_t lo4(int8_t x) {
    return (x & (1 << 3)) | (x & (1 << 2)) | (x & (1 << 1)) | (x & (1 << 0));
}

inline uint8_t hi4(uint8_t x) {
    return x >> 4;
}

inline uint8_t lo4(uint8_t x) {
    return x & 0x0F;
}

inline int8_t upc(int8_t h) {
    return h | (-((h & (1 << 3)) >> 3) & (-8));
}

typedef unsigned short ushort;
typedef unsigned int uint;

float half_to_float(const ushort x) {

    __m128i halfVector = _mm_cvtsi32_si128(x);
    __m128 floatVector = _mm_cvtph_ps(halfVector);
    return _mm_cvtss_f32(floatVector);
}

ushort float_to_half(const float x) {
    __m128 floatVector = _mm_set_ss(x);
    __m128i halfVector = _mm_cvtps_ph(floatVector, _MM_FROUND_TO_NEAREST_INT);
    return _mm_extract_epi16(halfVector, 0);
}

inline uint16_t int2hfloat(int8_t x)
{
    float inputFl32 = static_cast<float>(x);
    float* inputFl32_ptr = &inputFl32;
    unsigned int* fltInt32Ptr = reinterpret_cast<unsigned int*>(inputFl32_ptr);
    unsigned int fltInt32 = *fltInt32Ptr;
    unsigned short fltInt16;

    fltInt16 = (fltInt32 >> 31) << 5;
    unsigned short tmp = (fltInt32 >> 23) & 0xff;
    tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
    fltInt16 = (fltInt16 | tmp) << 10;
    fltInt16 |= (fltInt32 >> 13) & 0x3ff;

    return fltInt16;
}


void unpack(const int8_t* in, int8_t* out, int size) {
    for (int i = 0; i < size / 2; i++) {
        *(out++) = upc(lo4(*in));
        *(out++) = upc(hi4(*in));
        in++;
    }
}

void unpack_i4f16(const int8_t* in, int8_t* out, int size) {
    uint16_t *hFloatOut = reinterpret_cast<uint16_t *>(out);

    for (int i = 0; i < size / 2; i++) {
        *(hFloatOut++) = int2hfloat(upc(lo4(*in)));
        *(hFloatOut++) = int2hfloat(upc(hi4(*in)));
        in++;
    }
}

/*u4 order*/
void unpack_u4f32(const int8_t* in, float* out, int size) {
    for (int i = 0; i < size / 2; i++) {
        *(out++) = static_cast<float>(lo4(*in));
        *(out++) = static_cast<float>(hi4(*in));
        in++;
    }
}

template<typename T>
::testing::AssertionResult fp16ArraysMatch(const T &actual,
                                           const T &expected,
                                           const T &i4Input,
                                           bool int4 = 1 /*i4 or u4*/){
    for (size_t i = 0; i < expected.size() / 2; ++i) {

        int int8Input[] ={
                details::lo4(i4Input[i / 2]),
                details::hi4(i4Input[i / 2])
        };

        if (int4) {
            int8Input[0] = details::upc(int8Input[1]);
            int8Input[1] = details::upc(int8Input[0]);
        };

        auto fp16ref = int{*((uint16_t*)expected.data() + i)};
        auto fp16out = int{*((uint16_t*)actual.data() + i)};

#define _P(x) std::dec << std::setw(5) << (x) << '(' << std::setw(4) << std::hex << (x) << ')'
        if (fp16ref != fp16out) {
            return ::testing::AssertionFailure() << std::dec << std::setw(4) << i << ", i4:"
                                                 << std::setw(2) << int8Input[i % 2]
                                                 << " | ref " << _P(fp16ref)
                                                 << ", test "  << _P(fp16out) << "\n";
        }
#undef  _P

    }

    return ::testing::AssertionSuccess();
}

}  // namespace details

using ShapesInitializer = std::function<void (std::vector<int>&, std::vector<int>&, std::vector<int>&)>;


using UnpackTestsParams = std::tuple<
        ov::element::Type_t,  // fromPrecision
        ov::element::Type_t,  // toPrecision
        ov::element::Type_t,  // scalePrecision
        ov::element::Type_t,  // zeroPointPrecision
        unsigned long,        // nPartitions
        ShapesInitializer,    // input_shape , scale_shape, zerop initializer
        bool,                 // use parallel_for
        bool                  // strict partitioning
        >;

class UnpackTestsBase {
protected:
    ov::element::Type fromType;
    ov::element::Type toType;
    ov::element::Type scaleType;
    ov::element::Type zeropType;
    std::shared_ptr<ov::ITensor> from, to, scale, zerop;

    std::vector<int8_t> input;
    std::vector<int8_t> output;
    std::vector<int8_t> ref_output;
    std::vector<int8_t> scalesStorage;
    std::vector<int8_t> zeropStorage;
    float zeropValue;
    ov::Shape input_shape;
    ov::Shape scale_shape;
    ov::Shape zerop_shape;

    size_t nPartitions;
    bool useParallelFor = false;
    bool strictPartitions = false;

    void make_zeropoints() {
        if (zeropType == ov::element::undefined) {
            return;
        }

        const std::vector<float> zeropValues = {15.0f, 12.0f, 0.0f, 31.0f};
        const size_t nElements = shape_size(zerop_shape);

        // Set zeropValue if there's only one element
        if (nElements == 1) {
            zeropValue = zeropValues.front();
        }

        // Determine the size of the storage based on the type and resize the storage vector
        if (zeropType == ov::element::Type_t::u4) {
            zeropStorage.resize((nElements + 1) / 2, 0); // Each u4 zeropoint is 4 bits, so two zeropoints fit in one byte
        } else if (zeropType == ov::element::Type_t::f32) {
            zeropStorage.resize(nElements * sizeof(float), 0);
        } else {
            ASSERT_TRUE(zeropType == ov::element::u4 || zeropType == ov::element::f32);
        }

        // Fill the storage with the appropriate values
        if (zeropType == ov::element::Type_t::u4) {
            for (size_t i = 0; i < nElements; ++i) {
                uint8_t zeropValueU4 = static_cast<uint8_t>(zeropValues[i % zeropValues.size()]) & 0x0F;
                size_t byteIndex = i / 2;
                if (i % 2 == 0) {
                    zeropStorage[byteIndex] = zeropValueU4;
                } else {
                    zeropStorage[byteIndex] = (zeropValueU4 << 4);
                }
            }
        } else if (zeropType == ov::element::Type_t::f32) {
            float* ptrWork = reinterpret_cast<float*>(zeropStorage.data());
            for (size_t i = 0; i < nElements; ++i) {
                ptrWork[i] = zeropValues[i % zeropValues.size()];
            }
        }

        // Create the tensor
        zerop = ov::make_tensor(zeropType, zerop_shape, zeropStorage.data());
    }

    void make_scales() {
        if (scaleType == ov::element::undefined) {
            return;
        }
        ASSERT_TRUE(scaleType == ov::element::f16 || scaleType == ov::element::f32);
        size_t nElements = shape_size(scale_shape);

        // creating custom scale factors
        const size_t nScaleBytes  = scaleType.bitwidth() * nElements  / 8;

        std::vector<float> sc(nElements);
        float coeffTable[] = {
                0.1f,
                0.5f,
                1.f,
                2.f
        };
        for (size_t i = 0; i != nElements; i++) {
            sc[i] = coeffTable[i % (sizeof (coeffTable) / sizeof(*coeffTable))];
        }
        scalesStorage.resize(nScaleBytes);

        if (scaleType == ov::element::f16) {
            uint16_t * ptrWork = reinterpret_cast<uint16_t*>(scalesStorage.data());
            for (size_t i = 0; i != nElements; i++) {
                ptrWork[i] = details::float_to_half(sc[i]);
            }
        }
        if (scaleType == ov::element::f32) {
            float* ptrWork = reinterpret_cast<float*>(scalesStorage.data());
            for (size_t i = 0; i != nElements; i++) {
                ptrWork[i] = sc[i];
            }
        }
        scale = ov::make_tensor(scaleType, scale_shape, scalesStorage.data());
    }

    void make_input() {

        size_t nElements = shape_size(input_shape);

        ASSERT_EQ((fromType.bitwidth() * nElements) % 8, 0) << "Input len has to be byte boundary aligned, but was "
                                                            << fromType.bitwidth() * nElements << " bits";
        ASSERT_EQ((toType.bitwidth() * nElements) % 8, 0) << "Output len has to be byte boundary aligned";

        const size_t nInputBytes  = fromType.bitwidth() * nElements  / 8;
        const size_t nOutputBytes = toType.bitwidth() * nElements  / 8;

        input.resize(nInputBytes);
        ref_output.resize(nOutputBytes);
        output.resize(nOutputBytes);
        std::fill(ref_output.begin(), ref_output.end(), 0);
        std::fill(output.begin(), output.end(), 0);

        std::array<int8_t, 32> input_local = {
                0x0A, 0x0B, 0x1C, 0x1D, 0x2E, 0x2F, 0x35, 0x36,
                0x4A, 0x4B, 0x5A, 0x5B, 0x6A, 0x6B, 0x7A, 0x7B,
                0x0C, 0x0D, 0x1C, 0x1D, 0x2C, 0x2D, 0x3C, 0x3D,
                0x4C, 0x4D, 0x5C, 0x5D, 0x6C, 0x6D, 0x7C, 0x7D,
        };

        for (size_t idx = 0, k = 0; k < nInputBytes; k++, idx = (idx + 1) % input_local.size()) {
            input[k] = input_local[idx];
        }

        from = ov::make_tensor(fromType, input_shape, input.data());
        to = ov::make_tensor(toType, input_shape, output.data());
    }
public:
    void SetUp(const UnpackTestsParams & getParam) {
        ShapesInitializer shapeInit;

        std::tie(fromType, toType, scaleType, zeropType, nPartitions, shapeInit, useParallelFor, strictPartitions) = getParam;

        std::vector<int> input, scale, zerop;
        shapeInit(input, scale, zerop);

        input_shape = ov::Shape{input.begin(), input.end()};
        scale_shape = ov::Shape{scale.begin(), scale.end()};
        if (zerop.empty()) {
            zerop_shape = ov::Shape({1});
        } else {
            zerop_shape = ov::Shape{zerop.begin(), zerop.end()};
        }

        make_input();
        make_scales();
        make_zeropoints();

        make_ref_output();
    }
    std::string ToString() const {
        std::ostringstream result;
        result << (isNegative() ? "NEGATIVE_" : "")
               <<"[";

        for (size_t i = 0; i != input_shape.size(); i++) {
            result << input_shape[i] << ((i + 1 == input_shape.size()) ? "" : "x");
        }
        result <<"]"
               << "_p" << nPartitions
               << (strictPartitions ? "_SP" : "")
               << (useParallelFor ? "_parallel" : "_serial")
               << "_from_" << fromType
               << "_to_" << toType;
        if (scaleType != ov::element::Type_t::undefined)
            result << "_scale_" << scaleType;
        if (zeropType != ov::element::Type_t::undefined)
            result << "_zerop_" << zeropType;

        return result.str();
    }

    /**
     * Negative test cases has to be carefully reviewed, to still remain positive runs at some points
     * @return
     */
    virtual bool isNegative() const {
        return false;
    }

    virtual void make_ref_output() {
        size_t nElements = 1;
        for (size_t dim : input_shape) {
            nElements *= dim;
        }
        if (toType == ov::element::i8) {
            details::unpack(input.data(), ref_output.data(), static_cast<int>(nElements));
        } else if (toType == ov::element::f16) {
            details::unpack_i4f16(input.data(), ref_output.data(), static_cast<int>(nElements));
        }
    }
};

template <class T>
class UnpackTestsTmpl :
        public ::testing::Test,
        public T,
        public ::testing::WithParamInterface<UnpackTestsParams> {
protected:

    void SetUp() override {
        T::SetUp(GetParam());
    }
public:
    static std::string getTestCaseName(const testing::TestParamInfo<UnpackTestsParams>& obj) {
        T _bt;
        _bt.SetUp(obj.param);
        return _bt.ToString();
    }
};

using UnpackTests = UnpackTestsTmpl<UnpackTestsBase>;
class UnpackTestsRef : public UnpackTests {};

TEST_P(UnpackTests, i4) {
    ASSERT_NO_THROW_WITH_MESSAGE(ov::npuw::util::unpack(from, to, ov::npuw::util::UnpackOptions{useParallelFor, nPartitions, strictPartitions}));
    ASSERT_TRUE(details::fp16ArraysMatch(output, ref_output, input));
}

class UnpackWithScaleTestsBase : public UnpackTestsBase {
protected:
    bool isNegative() const override {
        if (scale_shape.size() != 3 && scale_shape.size() != 2) return true;
        if (input_shape.back() % 64) return true;
        if ((from->get_size() / scale->get_size()) % 64) return true;
        if (toType != ov::element::f16) return true;

        return false;
    }

    void make_ref_output() override {
        if (isNegative()) return;

        size_t nElements = from->get_size();

        const size_t nOutputElementsPerScale = ref_output.size() / (toType.bitwidth() / 8) / scale->get_size();

        details::unpack_i4f16(input.data(), ref_output.data(), static_cast<int>(nElements));

        // lets apply per channel scale
        uint16_t * pRef = reinterpret_cast<uint16_t*>(ref_output.data());
        uint16_t * pScale_f16 = reinterpret_cast<uint16_t*>(scale->data());
        float * pScale_f32 = reinterpret_cast<float*>(scale->data());

        for (size_t i = 0; i < scale->get_size(); i++) {
            for (size_t sc = 0; sc != nOutputElementsPerScale; sc++) {
                float ref_scaled = details::half_to_float(pRef[0]);
                if (scaleType == ov::element::f32) {
                    ref_scaled *= pScale_f32[0];
                } else if (scaleType == ov::element::f16) {
                    ref_scaled *= details::half_to_float(pScale_f16[0]);
                }
                *pRef = details::float_to_half(ref_scaled);
                pRef++;
            }
            pScale_f32++;
            pScale_f16++;
        }
    }

};

using UnpackWithScaleTests = UnpackTestsTmpl<UnpackWithScaleTestsBase>;


TEST_P(UnpackWithScaleTests, i4_scale) {
    ASSERT_NO_THROW_IF(!isNegative(),
                       ov::npuw::util::unpack(from, scale, to, ov::npuw::util::UnpackOptions{useParallelFor, nPartitions, strictPartitions}));
    if (!isNegative()) {
        ASSERT_TRUE(details::fp16ArraysMatch(output, ref_output, input));
    }
}


class UnpackTestsWithScaleAndZeroPointBase : public UnpackTestsBase {
protected:
    bool isNegative() const override {
        if (scale_shape.size() != 3 && scale_shape.size() != 2) return true;
        if (input_shape.back() % 64) return true;

        return false;
    }

    void make_ref_output() override {
        if (isNegative()) return;

        size_t nElements = from->get_size();

        const size_t nOutputElementsPerScale = ref_output.size() / (toType.bitwidth() / 8) / scale->get_size();

        std::vector<float> floatRef(nElements);
        details::unpack_u4f32(input.data(), floatRef.data(), static_cast<int>(nElements));


        // lets apply per channel scale
        uint16_t * pRef = reinterpret_cast<uint16_t*>(ref_output.data());
        float * pFloatRef = reinterpret_cast<float*>(floatRef.data());
        const uint16_t * pScale_f16 = reinterpret_cast<uint16_t*>(scale->data());
        const float * pScale_f32 = reinterpret_cast<float*>(scale->data());

        for (size_t i = 0; i < scale->get_size(); i++) {
            for (size_t sc = 0; sc != nOutputElementsPerScale; sc++) {
                // applying zeropoint
                float ref_scaled = *pFloatRef - zeropValue;

                if (scaleType == ov::element::f32) {
                    ref_scaled *= pScale_f32[0];
                } else if (scaleType == ov::element::f16) {
                    ref_scaled *= details::half_to_float(pScale_f16[0]);
                }
                *pRef = details::float_to_half(ref_scaled);

                pFloatRef++;
                pRef++;
            }
            pScale_f32++;
            pScale_f16++;
        }
    }
};

using UnpackTestsWithScaleAndZeroPoint = UnpackTestsTmpl<UnpackTestsWithScaleAndZeroPointBase>;

TEST_P(UnpackTestsWithScaleAndZeroPoint, u4) {
    ASSERT_NO_THROW_IF(!isNegative(),
                       ov::npuw::util::unpack(from, zerop, scale, to, ov::npuw::util::UnpackOptions{useParallelFor, nPartitions, strictPartitions}));
    if (!isNegative()) {
        ASSERT_TRUE(details::fp16ArraysMatch(output, ref_output, input, false));
    }
}

class UnpackTestsWithScaleAndZeroPoint2 : public UnpackTestsWithScaleAndZeroPointBase {
protected:
    bool isNegative() const override {
        if (input_shape.back() % 64 || input_shape.size() != 3) return true;
        if (scale_shape.back() % 64 || scale_shape.size() != 3) return true;

        return false;
    }

    void make_ref_output() override {
        if (isNegative()) return;

        size_t nElements = from->get_size();
        const auto from_shape = from->get_shape();

        const size_t C = from_shape[from_shape.size() - 3];
        const size_t H = from_shape[from_shape.size() - 2];
        const size_t W = from_shape[from_shape.size() - 1];

        std::vector<float> floatRef(nElements);
        details::unpack_u4f32(input.data(), floatRef.data(), static_cast<int>(nElements));

        uint16_t * pRef = reinterpret_cast<uint16_t*>(ref_output.data());
        float * pFloatRef = reinterpret_cast<float*>(floatRef.data());
        const uint16_t * pScale_f16 = reinterpret_cast<uint16_t*>(scale->data());
        const float * pScale_f32 = reinterpret_cast<float*>(scale->data());

        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t input_index =  w + W * h + W * H * c;
                    size_t scale_index = w + W * c;
                    float ref_scaled = pFloatRef[input_index] - zeropValue;
                    if (scaleType == ov::element::f32) {
                        ref_scaled *= pScale_f32[scale_index];
                    } else if (scaleType == ov::element::f16) {
                        ref_scaled *= details::half_to_float(pScale_f16[scale_index]);
                    }
                    pRef[w + W * h + c * W * H] = details::float_to_half(ref_scaled);
                }
            }
        }
    }
};

using UnpackTestsWithScaleAndZeroPointTest2 = UnpackTestsTmpl<UnpackTestsWithScaleAndZeroPoint2>;

TEST_P(UnpackTestsWithScaleAndZeroPointTest2, u4) {
    ASSERT_NO_THROW_IF(!isNegative(),
                       ov::npuw::util::unpack(from, zerop, scale, to, ov::npuw::util::UnpackOptions{useParallelFor, nPartitions, strictPartitions}));
    if (!isNegative()) {
        ASSERT_TRUE(details::fp16ArraysMatch(output, ref_output, input, false));
    }
}

class UnpackTestsWithScaleAndZeroPoint3 : public UnpackTestsWithScaleAndZeroPointBase {
protected:
    bool isNegative() const override {
        if (scale_shape.size() != 3 || zerop_shape.size() != 3) return true;
        if (input_shape[2] % 64 || input_shape.size() != 3) return true;

        return false;
    }

    void make_ref_output() override {
        if (isNegative()) return;

        size_t nElements = from->get_size();

        const size_t nOutputElementsPerScale = ref_output.size() / (toType.bitwidth() / 8) / scale->get_size();

        std::vector<float> floatRef(nElements);
        details::unpack_u4f32(input.data(), floatRef.data(), static_cast<int>(nElements));


        // lets apply per channel scale
        uint16_t * pRef = reinterpret_cast<uint16_t*>(ref_output.data());
        const uint8_t* pZer = static_cast<uint8_t*>(zerop->data());
        float * pFloatRef = reinterpret_cast<float*>(floatRef.data());
        const uint16_t * pScale_f16 = reinterpret_cast<uint16_t*>(scale->data());
        const float * pScale_f32 = reinterpret_cast<float*>(scale->data());

        for (size_t i = 0; i < scale->get_size(); i++) {
            float zeroPointValue = static_cast<float>((i % 2 == 0) ? details::lo4(pZer[i / 2]) : details::hi4(pZer[i / 2]));
            for (size_t sc = 0; sc != nOutputElementsPerScale; sc++) {
                // applying zeropoint
                float ref_scaled = *pFloatRef - zeroPointValue;

                if (scaleType == ov::element::f32) {
                    ref_scaled *= pScale_f32[0];
                } else if (scaleType == ov::element::f16) {
                    ref_scaled *= details::half_to_float(pScale_f16[0]);
                }
                *pRef = details::float_to_half(ref_scaled);

                pFloatRef++;
                pRef++;
            }
            pScale_f32++;
            pScale_f16++;
        }
    }
};

using UnpackTestsWithScaleAndZeroPointTest3 = UnpackTestsTmpl<UnpackTestsWithScaleAndZeroPoint3>;

TEST_P(UnpackTestsWithScaleAndZeroPointTest3, u4) {
    ASSERT_NO_THROW_IF(!isNegative(),
                       ov::npuw::util::unpack(from, zerop, scale, to, ov::npuw::util::UnpackOptions{useParallelFor, nPartitions, strictPartitions}));
    if (!isNegative()) {
        ASSERT_TRUE(details::fp16ArraysMatch(output, ref_output, input, false));
    }
}

#define Tensors [](std::vector<int>& input, std::vector<int>&scale, std::vector<int>&zerop)


namespace details {
::testing::internal::ParamGenerator<typename std::vector<ShapesInitializer>::value_type> ShapesIn(
        const std::vector<ShapesInitializer>& container) {
    return ::testing::ValuesIn(container.begin(), container.end());
}

}  // namespace details
}  // anonymous namespace
