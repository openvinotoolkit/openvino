// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <functional>
#include <algorithm>
#include <string>
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

PRETTY_PARAM(NDims, nd_tensor_test_params);

auto refSelect = [](const float a, const float b, const float c) noexcept {
    return (a != 0) ? b : c;
};

typedef float (*kernel)(const float a, const float b, const float c);

void genRandomDataLogic(Blob::Ptr blob);
void getCoord(uint32_t nSubspace, SizeVector dims, uint32_t subspaceCoord[]);
int getNum(uint32_t subspaceDims[], SizeVector dims);
SizeVector convertDims(SizeVector dims);

class myriadLayersTestsSelectBase: public myriadLayersTests_nightly {
protected:
    void RefSelect()
    {
        auto itr = _inputMap.begin();
        int coeff_num = 0;
        const uint16_t *srcData = itr->second->buffer().as<const uint16_t*>();
        uint16_t *dstData = _refBlob->buffer().as<uint16_t*>();
        uint32_t src_coords[4];
        SizeVector refDims = convertDims(_refBlob->getTensorDesc().getDims());
        SizeVector itrDims = convertDims(itr->second->getTensorDesc().getDims());

        itr++;
        ASSERT_NE(itr, _inputMap.end());
        const uint16_t *src1Data = itr->second->buffer().as<const uint16_t*>();
        SizeVector itr1Dims = convertDims(itr->second->getTensorDesc().getDims());
        itr++;
        ASSERT_NE(itr, _inputMap.end());
        const uint16_t *src2Data = itr->second->buffer().as<const uint16_t*>();
        SizeVector itr2Dims = convertDims(itr->second->getTensorDesc().getDims());
        itr++;
        ASSERT_EQ(itr, _inputMap.end());

        for (int i = 0; i < _refBlob->size(); i++) {
            getCoord(i, refDims, src_coords);

            uint32_t src1_coords[4], src2_coords[4];
            for (int c = 0; c < refDims.size(); c++) {
                src2_coords[c] = src1_coords[c] = src_coords[c];
                if (src_coords[c] >= itrDims[c])
                    src_coords[c] = 0;
                if (src1_coords[c] >= itr1Dims[c])
                    src1_coords[c] = 0;
                if (src2_coords[c] >= itr2Dims[c])
                    src2_coords[c] = 0;
            }

            int src_i = getNum(src_coords, itrDims);
            int src1_i = getNum(src1_coords, itr1Dims);
            int src2_i = getNum(src2_coords, itr2Dims);

            float val = refSelect(PrecisionUtils::f16tof32(srcData[src_i]),
                                  PrecisionUtils::f16tof32(src1Data[src1_i]),
                                  PrecisionUtils::f16tof32(src2Data[src2_i]));
            dstData[i] = PrecisionUtils::f32tof16(val);
        }
    }

    nd_tensor_test_params _p;
    std::map<std::string, std::string> _params;

};

class SelectTest : public myriadLayersTestsSelectBase,
                   public testing::WithParamInterface<std::tuple<NDims, int>> {
protected:
    virtual void InitBody()
    {
        float ERROR_BOUND;

        ERROR_BOUND = 8.4e-3f;

        _params.clear();
        auto params = GetParam();
        _p = std::get<0>(params);
        int ndims = std::get<1>(params);
        int count = 3; // Select support only 3 onputs

        InferenceEngine::SizeVector dims;
        dims.resize(ndims);
        for (int i = 0; i < ndims; i++)
            dims[i] = _p.dims[i];

        IN_OUT_desc inpt(count);
        for (int i = 0; i < count; ++i) {
            inpt[i] = dims;
        }

        SetInputTensors(inpt);
        SetOutputTensors({dims});

        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Select").params(_params)));
        ASSERT_TRUE(Infer());

        ASSERT_NO_FATAL_FAILURE(RefSelect());
        ASSERT_EQ(_outputMap.size(), 1);

        CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
    }
};

class myriadTestsSelect_smoke: public SelectTest
{
    void SetUp() override {
        SelectTest::SetUp();
        _genDataCallback0 = genRandomDataLogic;
    }
};

TEST_P(myriadTestsSelect_smoke, Select)
{
    InitBody();
}

static std::vector<NDims> s_eltwiseTensors = {
        {{3, 2, 14, 32}},
        {{5, 4, 8, 16}},
        {{2, 16, 16, 8}},
};

static std::vector<int> s_eltwiseDims = {
        2, 3, 4
};
