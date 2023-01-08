// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

using myriadConcatTestParams = std::tuple<InferenceEngine::SizeVector, int32_t, InferenceEngine::SizeVector, int32_t, int32_t >;
typedef myriadLayerTestBaseWithParam<myriadConcatTestParams> myriadLayersTestsConcat_smoke;

void CheckOutput(const InferenceEngine::BlobMap& input, InferenceEngine::Blob::Ptr actual, int32_t axis) {
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;

    get_ndims(actual, OW, OH, OC, ON);

    int32_t OFFSET[3] = {};
    auto actual_data = actual->buffer().as<const uint16_t*>();
    int input_idx = 0;
    int n_checks = 0;

    for (auto inputElem : input) {
        int32_t INP[4] = {};
        get_ndims(inputElem.second, INP[0], INP[1], INP[2], INP[3]);
        auto src_data = inputElem.second->buffer().as<const uint16_t*>();
        size_t output_size =  OW * OH * OC;
        size_t input_size =  INP[0] * INP[1] * INP[2];
        for (int32_t n = 0; n < INP[3]; ++n) {
            for (int32_t h = 0; h < INP[1]; ++h) {
                for (int32_t w = 0; w < INP[0]; ++w) {
                    for (int32_t c = 0; c < INP[2]; ++c) {
                        n_checks++;
                        size_t oodx = c + OFFSET[2] + OC * ((w + OFFSET[0]) + (h + OFFSET[1]) * OW)  +  n * output_size;
                        size_t iidx = c + INP[2] * (w + h * INP[0]) + n * input_size;
                        ASSERT_EQ(actual_data[oodx], src_data[iidx])
                                    << "at: input=" << input_idx << " n=" << n << " c=" << c << " h=" << h << " w=" << w
                                    << ", actual data : " << PrecisionUtils::f16tof32(actual_data[oodx])
                                    << " reference data " << PrecisionUtils::f16tof32(src_data[iidx]);
                    }
                }
            }
        }
        OFFSET[axis] += INP[axis];
        input_idx++;
    }
    ASSERT_NE(n_checks, 0);
}

TEST_P(myriadLayersTestsConcat_smoke, Concat) {
    auto param   = GetParam();
    auto core    = std::get<0>(param);
    auto axis    = std::get<1>(param);
    auto shifts  = std::get<2>(param);
    auto numDims = std::get<3>(param);
    auto batch   = std::get<4>(param);

    ASSERT_EQ(core.size(), 2);
    axis %= numDims;
    IN_OUT_desc dims;
    IN_OUT_desc output(1);
    output[0].resize(numDims);

    int32_t channelsSum = 0;
    uint32_t offset0 = 0;
    uint32_t offset1 = 0;
    int32_t shifted_axis = numDims - 1 - axis;
    switch(numDims) {
        case 4:
            offset0 = 1 + ((axis) % 3);
            offset1 = 1 + ((axis + 1) % 3);
            for (auto elem : shifts) {
                InferenceEngine::SizeVector newSlice(numDims);
                newSlice[0] = batch;
                newSlice[axis] = elem;
                newSlice[offset0] = core[0];
                newSlice[offset1] = core[1];
                channelsSum += elem;
                dims.push_back(newSlice);
            }
            output[0][0] = batch;
            output[0][offset1] = core[1];
            break;
        case 2:
            shifted_axis = (batch == 1 ? 2 : 3)  - 1 - axis;
            offset0 = 1 + ((axis + 1) % (numDims));
            axis++;
            for (auto elem : shifts) {
                InferenceEngine::SizeVector newSlice(batch == 1 ? 3 : 4, 1);
                newSlice[0] = batch;
                newSlice[axis] = elem;
                newSlice[offset0] = core[0];
                dims.push_back(newSlice);
                channelsSum += elem;
            }
            output[0].resize(batch == 1 ? 3 : 4, 1);
            output[0][0] = batch;
            break;
        case 1:
            offset0 = 1 + ((axis + 1) % (numDims));
            axis++;
            for (auto elem : shifts) {
                InferenceEngine::SizeVector newSlice(numDims + 1);
                newSlice[0] = batch;
                newSlice[axis] = elem;
                //newSlice[offset0] = core[0];
                channelsSum += elem;
                dims.push_back(newSlice);
            }
            output[0].resize(numDims + 1);
            output[0][0] = batch;
            break;
        default:
            FAIL() << "Unsupported tensor dimension.";
    }
    output[0][axis] = channelsSum;
    if (numDims > 1) {
        output[0][offset0] = core[0];
    }

    SetInputTensors(dims);
    SetOutputTensors(output);
    std::map<std::string, std::string> params;
    params["axis"] = std::to_string(axis);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Concat").params(params), NetworkInitParams().layoutPreference(vpu::LayoutPreference::ChannelMinor)));
    ASSERT_TRUE(Infer());
    auto dst = _outputMap.begin()->second;
    CheckOutput(_inputMap, dst, shifted_axis);
}

static  std::vector<int32_t> s_axis = {
    1, 2, 3
};

static  std::vector<int32_t> s_dimension = {
    1, 2, 4
};

static  std::vector<int32_t> s_batch = {
    1 /*, 8 TODO: rewrite to ngraph to have reshape functionality */
};

static std::vector<InferenceEngine::SizeVector> s_concatCores = {
    {{8, 4}, { 8, 16}, {8, 8}}
};

static std::vector<InferenceEngine::SizeVector> s_concatInputs = {
    {{1,}, {1, 2, 4}, {1, 2, 3, 4, 5}, {2, 4}}
};

//function is returning correct name to gtest
std::string getTestCaseName(testing::TestParamInfo<myriadConcatTestParams> param) {
    auto core    = std::get<0>(param.param);
    auto axis    = std::get<1>(param.param);
    auto shifts  = std::get<2>(param.param);
    auto numDims = std::get<3>(param.param);
    auto batch   = std::get<4>(param.param);

    std::stringstream ss;
    ss<<"core="<<core<<"/axis="<<axis<<"/shifts="<<shifts<<"/numDims="<<(numDims!=4 ? numDims+1 : numDims)<<"/batch="<<batch;
    return ss.str();
}
