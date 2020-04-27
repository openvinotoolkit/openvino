// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <functional>
#include <algorithm>
#include <string>
#include "myriad_layers_reference_functions.hpp"

// TODO: no tests for multiple inputs to eltwise at all
extern const char ELTWISE_MAX[] = "max";
extern const char ELTWISE_MUL[] = "mul";
extern const char ELTWISE_SUM[] = "sum";
extern const char ELTWISE_SUB[] = "sub";
extern const char ELTWISE_DIV[] = "div";
extern const char ELTWISE_MIN[] = "min";
extern const char ELTWISE_SQDIFF[] = "squared_diff";
extern const char ELTWISE_POW[] = "pow";
extern const char ELTWISE_FLOOR_MOD[] = "floor_mod";
extern const char ELTWISE_EQUAL[] = "equal";
extern const char ELTWISE_NOT_EQUAL[] = "not_equal";
extern const char ELTWISE_GREATER[] = "greater";
extern const char ELTWISE_GREATER_EQUAL[] = "greater_equal";
extern const char ELTWISE_LESS[] = "less";
extern const char ELTWISE_LESS_EQUAL[] = "less_equal";
extern const char ELTWISE_LOGICAL_NOT[] = "logical_not";
extern const char ELTWISE_LOGICAL_AND[] = "logical_and";
extern const char ELTWISE_LOGICAL_OR[] = "logical_or";
extern const char ELTWISE_LOGICAL_XOR[] = "logical_xor";
extern const char ELTWISE_MEAN[] = "mean";

using namespace InferenceEngine;

PRETTY_PARAM(NDims, nd_tensor_test_params);

auto refMax = [](const float a, const float b, const float /*c*/)noexcept {
    return std::max(a, b);
};

auto refMul = [](const float a, const float b, const float /*c*/)noexcept {
    return a * b;
};

auto refSum = [](const float a, const float b, const float /*c*/)noexcept {
    return a + b;
};

auto refSub = [](const float a, const float b, const float /*c*/) noexcept {
    return a - b;
};

auto refDiv = [](const float a, const float b, const float /*c*/) noexcept {
    return a / b;
};

auto refMin = [](const float a, const float b, const float /*c*/) noexcept {
    return std::min(a, b);
};

auto refSqDiff = [](const float a, const float b, const float /*c*/) noexcept {
    return (a - b) * (a - b);
};

auto refPow = [](const float a, const float b, const float /*c*/) noexcept {
    return powf(a, b);
};

auto refFloorMod = [](const float a, const float b, const float /*c*/) noexcept {
    return a - b * floorf(a / b);
};

auto refEqual = [](const float a, const float b, const float /*c*/) noexcept {
    return a == b ? 1.f : 0.f;
};

auto refNotEqual = [](const float a, const float b, const float /*c*/) noexcept {
    return a != b ? 1.f : 0.f;
};

auto refGreater = [](const float a, const float b, const float /*c*/) noexcept {
    return a > b ? 1.f : 0.f;
};

auto refGreaterEqual = [](const float a, const float b, const float /*c*/) noexcept {
    return a >= b ? 1.f : 0.f;
};

auto refLess = [](const float a, const float b, const float /*c*/) noexcept {
    return a < b ? 1.f : 0.f;
};

auto refLessEqual = [](const float a, const float b, const float /*c*/) noexcept {
    return a <= b ? 1.f : 0.f;
};

auto refLogicalNot = [](const float a, const float b, const float /*c*/) noexcept {
    return (a == 0) ? 1.f : 0.f;
};

auto refLogicalAnd = [](const float a, const float b, const float /*c*/) noexcept {
    return (a != 0) && (b != 0) ? 1.f : 0.f;
};

auto refLogicalOr = [](const float a, const float b, const float /*c*/) noexcept {
    return (a != 0) || (b != 0) ? 1.f : 0.f;
};

auto refLogicalXor = [](const float a, const float b, const float /*c*/) noexcept {
    return int((a != 0) && !(b != 0)) + int(!(a != 0) && (b != 0)) ? 1.f : 0.f;
};

auto refMean = [](const float a, const float b, const float /*c*/) noexcept {
    return (a + b)/2.f;
};

typedef float (*kernel)(const float a, const float b, const float c);

static const std::map<const char*, kernel> s_kernels = {
        {ELTWISE_MAX, refMax},
        {ELTWISE_MUL, refMul},
        {ELTWISE_SUM, refSum},
        {ELTWISE_SUB, refSub},
        {ELTWISE_DIV, refDiv},
        {ELTWISE_MIN, refMin},
        {ELTWISE_SQDIFF, refSqDiff},
        {ELTWISE_POW, refPow},
        {ELTWISE_FLOOR_MOD, refFloorMod},
        {ELTWISE_EQUAL, refEqual},
        {ELTWISE_NOT_EQUAL, refNotEqual},
        {ELTWISE_GREATER, refGreater},
        {ELTWISE_GREATER_EQUAL, refGreaterEqual},
        {ELTWISE_LESS, refLess},
        {ELTWISE_LESS_EQUAL, refLessEqual},
        {ELTWISE_LOGICAL_NOT, refLogicalNot},
        {ELTWISE_LOGICAL_AND, refLogicalAnd},
        {ELTWISE_LOGICAL_OR, refLogicalOr},
        {ELTWISE_LOGICAL_XOR, refLogicalXor},
        {ELTWISE_MEAN, refMean}
};

void genRandomDataPow(Blob::Ptr blob) {
    float scale = 2.0f / RAND_MAX;
    /* fill by random data in the range (-1, 1)*/
    auto * blobRawDataFp16 = blob->buffer().as<ie_fp16 *>();
    size_t count = blob->size();
    for (size_t indx = 0; indx < count; ++indx) {
        float val = rand();
        val = val * scale - 1.0f;
        while (fabs(val) < .01f) {
            val *= 10.f;
        }
        blobRawDataFp16[indx] = PrecisionUtils::f32tof16(val);
    }
}

void genRandomDataLogic(Blob::Ptr blob) {
    /*fill inputs by 0x0000 or 0xFFFF*/
    auto * blobRawDataFp16 = blob->buffer().as<ie_fp16 *>();
    size_t count = blob->size();
    const auto TrueVal = PrecisionUtils::f32tof16(1.f);
    const auto FalseVal = PrecisionUtils::f32tof16(0.f);
    float scale = 1.0f / RAND_MAX;
    for (size_t indx = 0; indx < count; ++indx) {
        float val = rand() * scale;
        blobRawDataFp16[indx] = val <.5f ? FalseVal : TrueVal;
    }
}

void getCoord(uint32_t nSubspace, SizeVector dims, uint32_t subspaceCoord[])
{
    for(int i = 0; i < dims.size(); ++i) {
        int nUpSubspace = nSubspace / dims[i];
        subspaceCoord[i] = nSubspace - nUpSubspace * dims[i];
        nSubspace = nUpSubspace;
    }
}

int getNum(uint32_t subspaceDims[], SizeVector dims)
{
    int totalSubspaces = 1;
    int num = 0;
    for(int i = 0; i < dims.size(); i++) {
        num += totalSubspaces * subspaceDims[i];
        totalSubspaces *= dims[i];
    }
    return num;
}

SizeVector convertDims(SizeVector dims)
{
    SizeVector ret(4);
    if (dims.size() == 1) {
        ret[0] = dims[0];
        ret[1] = 1;
        ret[2] = 1;
        ret[3] = 1;
        return ret;
    }

    if (dims.size() == 2) {
        ret[0] = dims[1];
        ret[1] = 1;
        ret[2] = 1;
        ret[3] = dims[0];
        return ret;
    }

    if (dims.size() == 3) {
        ret[0] = dims[0];
        ret[1] = dims[2];
        ret[2] = dims[1];
        ret[3] = 1;
        return ret;
    }

    else {// (dims.size() == 4)
        ret[0] = dims[1];
        ret[1] = dims[3];
        ret[2] = dims[2];
        ret[3] = dims[0];
        return ret;
    }
}

class myriadLayersTestsEltwiseBase: public myriadLayersTests_nightly {
protected:
    template <typename Func>void RefEltwise(Func fun, std::vector<float> coeff)
    {
        auto itr = _inputMap.begin();
        int coeff_num = 0;
        const uint16_t *srcData = itr->second->buffer().as<const uint16_t*>();
        uint16_t *dstData = _refBlob->buffer().as<uint16_t*>();
        uint32_t src_coords[4];
        SizeVector refDims = convertDims(_refBlob->getTensorDesc().getDims());
        SizeVector itrDims = convertDims(itr->second->getTensorDesc().getDims());

        if (fun == s_kernels.at(ELTWISE_LOGICAL_NOT)) {
            for (int i = 0; i < _refBlob->size(); i++) {
                getCoord(i, refDims, src_coords);

                for (int c = 0; c < refDims.size(); c++)
                    if (src_coords[c] >= itrDims[c])
                        src_coords[c] = 0;

                int src_i = getNum(src_coords, itrDims);

                dstData[i] = PrecisionUtils::f32tof16(fun(PrecisionUtils::f16tof32(srcData[src_i]), 0.f, 0.f));
            }
        } else {
            for (int i = 0; i < _refBlob->size(); i++) {
                getCoord(i, refDims, src_coords);

                for (int c = 0; c < refDims.size(); c++)
                    if (src_coords[c] >= itrDims[c])
                        src_coords[c] = 0;

                int src_i = getNum(src_coords, itrDims);

                dstData[i] = PrecisionUtils::f32tof16(PrecisionUtils::f16tof32(srcData[src_i]) * coeff[coeff_num]);
            }
        }

        itr++;
        coeff_num++;

        while(itr != _inputMap.end()) {
            ASSERT_NE(itr->second, nullptr);
            const uint16_t *srcData = itr->second->buffer().as<const uint16_t*>();
            ASSERT_NE(srcData, nullptr);
            uint16_t *dstData = _refBlob->buffer().as<uint16_t*>();
            itrDims = convertDims(itr->second->getTensorDesc().getDims());

            for (int i = 0; i < _refBlob->size(); i++) {
                getCoord(i, refDims, src_coords);

                for (int c = 0; c < refDims.size(); c++)
                    if (src_coords[c] >= itrDims[c])
                        src_coords[c] = 0;

                int src_i = getNum(src_coords, itrDims);
                float val = fun(PrecisionUtils::f16tof32(dstData[i]), PrecisionUtils::f16tof32(srcData[src_i])*coeff[coeff_num], 0.f);

                dstData[i] = PrecisionUtils::f32tof16(val);
            }
            ++itr;
            ++coeff_num;
        }
    }

    nd_tensor_test_params _p;
    std::map<std::string, std::string> _params;

};

template <const char* EltwiseType> class EltwiseTest : public myriadLayersTestsEltwiseBase,
                                                       public testing::WithParamInterface<std::tuple<NDims, int, int>> {
protected:
    virtual void InitBody(bool withCoefs = false, bool withBroadcast = false, bool isOutputLogic = false)
    {
        float ERROR_BOUND;

        if (strcmp(EltwiseType, ELTWISE_POW) == 0)
            ERROR_BOUND = .125f;
        else
            ERROR_BOUND = 8.4e-3f;

        _params.clear();
        auto params = GetParam();
        _p = std::get<0>(params);
        int count = std::get<1>(params);
        int ndims = std::get<2>(params);

        _params["operation"] = EltwiseType;

        std::vector<float> coeff;
        for (int i = 0; i < count; i++)
            coeff.push_back(withCoefs ? ((float)rand() / RAND_MAX) * 2.0f : 1.0f);
        if (withCoefs) {
            _params["coeff"] = std::to_string(coeff[0]);
            for (int i = 1; i < count; i++)
                _params["coeff"] += "," + std::to_string(coeff[i]);
        }

        InferenceEngine::SizeVector dims;
        dims.resize(ndims);
        for (int i = 0; i < ndims; i++)
            dims[i] = _p.dims[i];

        IN_OUT_desc inpt(count);
        for (int i = 0; i < count; ++i) {
            inpt[i] = dims;
        }

        if (withBroadcast) {
            if(ndims == 3) {
                GTEST_SKIP_("Please look at #-19681");
                // inpt[2].resize(2);
            } else {
                inpt[rand()%count].resize(rand()%ndims + 1);
            }
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < inpt[i].size(); j++) {
                    if (rand()%2 > 0) {
                        inpt[i][j] = 1;
                    }
                }
            }
        }

        SetInputTensors(inpt);
        SetOutputTensors({dims});

        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

        ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Eltwise").params(_params)));
        ASSERT_TRUE(Infer());

        ASSERT_NO_FATAL_FAILURE(RefEltwise(s_kernels.at(EltwiseType), coeff));
        ASSERT_EQ(_outputMap.size(), 1);

        if (isOutputLogic) {
            Blob::Ptr& output = _outputMap.begin()->second;
            size_t out_size = output->size();
            InferenceEngine::ie_fp16 *output_fp16_ptr = output->buffer().as<ie_fp16*>();

            for (size_t i = 0; i < out_size; i++) {
                if (PrecisionUtils::f16tof32(output_fp16_ptr[i]) != 0.f) {
                    output_fp16_ptr[i] = PrecisionUtils::f32tof16(1.f);
                }
            }
        }

        CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
    }
};

class myriadTestsEltwiseMax_nightly: public EltwiseTest<ELTWISE_MAX>
{
};

class myriadTestsEltwiseSum_nightly: public EltwiseTest<ELTWISE_SUM>
{
};

class myriadTestsEltwiseSub_nightly: public EltwiseTest<ELTWISE_SUB>
{
};

class myriadTestsEltwiseMul_nightly: public EltwiseTest<ELTWISE_MUL>
{
};

class myriadTestsEltwiseSumWithCoeff_nightly: public EltwiseTest<ELTWISE_SUM>
{
};

class myriadTestsEltwiseSubWithCoeff_nightly: public EltwiseTest<ELTWISE_SUB>
{
};

class myriadTestsEltwiseSumWithBroadcast_nightly: public EltwiseTest<ELTWISE_SUM>
{
};

class myriadTestsEltwiseSubWithBroadcast_nightly: public EltwiseTest<ELTWISE_SUB>
{
};

class myriadTestsEltwiseDiv_nightly: public EltwiseTest<ELTWISE_DIV>
{
};

class myriadTestsEltwiseMin_nightly: public EltwiseTest<ELTWISE_MIN>
{
};

class myriadTestsEltwiseSqDiff_nightly: public EltwiseTest<ELTWISE_SQDIFF>
{
};

class myriadTestsEltwisePow_nightly: public EltwiseTest<ELTWISE_POW>
{
    void SetUp() override {
        EltwiseTest::SetUp();
        _genDataCallback = genRandomDataPow;
    }
};

class myriadTestsEltwiseFloorMod_nightly: public EltwiseTest<ELTWISE_FLOOR_MOD>
{
};

class myriadTestsEltwiseEqual_nightly: public EltwiseTest<ELTWISE_EQUAL>
{
};

class myriadTestsEltwiseNotEqual_nightly: public EltwiseTest<ELTWISE_NOT_EQUAL>
{
};

class myriadTestsEltwiseGreater_nightly: public EltwiseTest<ELTWISE_GREATER>
{
};

class myriadTestsEltwiseGreaterEqual_nightly: public EltwiseTest<ELTWISE_GREATER_EQUAL>
{
};

class myriadTestsEltwiseLess_nightly: public EltwiseTest<ELTWISE_LESS>
{
};

class myriadTestsEltwiseLessEqual_nightly: public EltwiseTest<ELTWISE_LESS_EQUAL>
{
};

class myriadTestsEltwiseLogicalNot_nightly: public EltwiseTest<ELTWISE_LOGICAL_NOT>
{
    void SetUp() override {
        EltwiseTest::SetUp();
        _genDataCallback = genRandomDataLogic;
    }
};

class myriadTestsEltwiseLogicalAnd_nightly: public EltwiseTest<ELTWISE_LOGICAL_AND>
{
    void SetUp() override {
        EltwiseTest::SetUp();
        _genDataCallback = genRandomDataLogic;
    }
};

class myriadTestsEltwiseLogicalOr_nightly: public EltwiseTest<ELTWISE_LOGICAL_OR>
{
    void SetUp() override {
        EltwiseTest::SetUp();
        _genDataCallback = genRandomDataLogic;
    }
};

class myriadTestsEltwiseLogicalXor_nightly: public EltwiseTest<ELTWISE_LOGICAL_XOR>
{
    void SetUp() override {
        EltwiseTest::SetUp();
        _genDataCallback = genRandomDataLogic;
    }
};

class myriadTestsEltwiseMean_nightly: public EltwiseTest<ELTWISE_MEAN>
{
};

TEST_P(myriadTestsEltwiseMax_nightly, Max)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseSum_nightly, Sum)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseSub_nightly, Sub)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseMul_nightly, Mul)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseSumWithCoeff_nightly, Sum)
{
    InitBody(true);
}

TEST_P(myriadTestsEltwiseSubWithCoeff_nightly, Sub)
{
    InitBody(true);
}

TEST_P(myriadTestsEltwiseSumWithBroadcast_nightly, Sum)
{
    InitBody(false, true);
}

TEST_P(myriadTestsEltwiseSubWithBroadcast_nightly, Sub)
{
    InitBody(false, true);
}

TEST_P(myriadTestsEltwiseDiv_nightly, Div)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseMin_nightly, Min)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseSqDiff_nightly, SqDiff)
{
    InitBody();
}

TEST_P(myriadTestsEltwisePow_nightly, Pow)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseFloorMod_nightly, FloorMod)
{
    InitBody();
}

TEST_P(myriadTestsEltwiseEqual_nightly, Equal)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseNotEqual_nightly, NotEqual)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseGreater_nightly, Greater)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseGreaterEqual_nightly, GreaterEqual)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseLess_nightly, Less)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseLessEqual_nightly, LessEqual)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseLogicalNot_nightly, LogicalNot)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseLogicalAnd_nightly, LogicalAnd)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseLogicalOr_nightly, LogicalOr)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseLogicalXor_nightly, LogicalXor)
{
    InitBody(false, false, true);
}

TEST_P(myriadTestsEltwiseMean_nightly, Mean)
{
    InitBody();
}

static std::vector<NDims> s_eltwiseTensors = {
        {{3, 2, 14, 32}},
        {{5, 4, 8, 16}},
        {{2, 16, 16, 8}},
};

static std::vector<int> s_eltwiseInputs = {
        2, 3, 4, 5, 6
};

static std::vector<int> s_eltwiseOnlyTwoInputs = {
        2
};

static std::vector<int> s_eltwiseOnlyOneInput = {
        1
};

static std::vector<int> s_eltwiseDims = {
        2, 3, 4
};
