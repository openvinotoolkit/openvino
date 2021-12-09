// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <cmath>

#define BOUND (5.0f)

using namespace InferenceEngine;

struct strided_slice_test_param {
    InferenceEngine::SizeVector in_shape;
    size_t dim_size;
    std::vector<int32_t> begin;
    std::vector<int32_t> end;
    std::vector<int32_t> strides;

    InferenceEngine::SizeVector begin_mask;
    InferenceEngine::SizeVector end_mask;
    InferenceEngine::SizeVector ellipsis_mask;
    InferenceEngine::SizeVector new_axis_mask;
    InferenceEngine::SizeVector shrink_axis_mask;
    InferenceEngine::SizeVector out_shape;
};

class myriadLayersTestsStridedSlice_smoke: public myriadLayersTests_nightly,
                                           public testing::WithParamInterface<strided_slice_test_param> {
public:
    std::string model_t = R"V0G0N(
<net Name="StridedSlice_net" version="2" precision="FP16" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP16" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="begin" type="Const" precision="I32" id="2">
            <output>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_DIM_BYTE_OFFSET_0_" size="_DIM_BYTE_SIZE_0_"/>
            </blobs>
        </layer>
        <layer name="end" type="Const" precision="I32" id="3">
            <output>
                <port id="3">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_DIM_BYTE_OFFSET_1_" size="_DIM_BYTE_SIZE_1_"/>
            </blobs>
        </layer>
        _STRIDES_IN_LAYER_
        <layer name="strided_slice" id="5" type="StridedSlice" precision="FP16">
            <data _BEGIN_ _END_ _ELLIPSIS_ _NEW_AXIS_ _SHRINK_/>
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
                <port id="3">
                    <dim>_DIM_SIZE_</dim>
                </port>
                _STRIDES_IN_PORT_
            </input>
            <output>
                <port id="5">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="5" to-port="3"/>
        _STRIDES_IN_EDGE_
    </edges>
</net>
)V0G0N";

std::string stridesLayer = R"V0G0N(
<layer name="strides" type="Const" precision="I32" id="4">
    <output>
        <port id="4">
            <dim>_DIM_SIZE_</dim>
        </port>
    </output>
    <blobs>
        <custom offset="_DIM_BYTE_OFFSET_2_" size="_DIM_BYTE_SIZE_2_"/>
    </blobs>
</layer>
)V0G0N";

std::string stridesInPort = R"V0G0N(
<port id="4">
    <dim>_DIM_SIZE_</dim>
</port>
)V0G0N";

std::string stridesEdge = R"V0G0N(
<edge from-layer="4" from-port="4" to-layer="5" to-port="4"/>
)V0G0N";

    std::string getModel(const strided_slice_test_param& p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape;
        std::string begin;
        std::string end;
        std::string ellipsis;
        std::string new_axis;
        std::string shrink_axis;

        for (const auto& i : p.in_shape) {
            in_shape += "<dim>";
            in_shape += std::to_string(i) + "</dim>\n";
        }
        in_shape.pop_back();

        if (!p.strides.empty()) {
            REPLACE_WITH_NUM(stridesLayer, "_DIM_BYTE_SIZE_2_", p.strides.size() * sizeof(uint32_t));
            REPLACE_WITH_NUM(stridesLayer, "_DIM_BYTE_OFFSET_2_", (p.begin.size() + p.end.size()) * sizeof(uint32_t));
            REPLACE_WITH_NUM(model, "_STRIDES_IN_LAYER_", stridesLayer);
            REPLACE_WITH_NUM(model, "_STRIDES_IN_PORT_", stridesInPort);
            REPLACE_WITH_NUM(model, "_STRIDES_IN_EDGE_", stridesEdge);
        } else {
            REPLACE_WITH_NUM(model, "_STRIDES_IN_LAYER_", std::string());
            REPLACE_WITH_NUM(model, "_STRIDES_IN_PORT_", std::string());
            REPLACE_WITH_NUM(model, "_STRIDES_IN_EDGE_", std::string());
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.dim_size);
        REPLACE_WITH_NUM(model, "_DIM_BYTE_SIZE_0_", p.begin.size() * sizeof(uint32_t));
        REPLACE_WITH_NUM(model, "_DIM_BYTE_SIZE_1_", p.end.size() * sizeof(uint32_t));
        REPLACE_WITH_NUM(model, "_DIM_BYTE_OFFSET_0_", 0);
        REPLACE_WITH_NUM(model, "_DIM_BYTE_OFFSET_1_", p.begin.size() * sizeof(uint32_t));

        if (!p.begin_mask.empty()) {
            begin = "begin_mask=\"";
            for (const auto& pb : p.begin_mask)
                begin += std::to_string(pb) + ",";
            begin.pop_back();
            begin += "\"";
        }
        REPLACE_WITH_STR(model, "_BEGIN_", begin);

        if (!p.end_mask.empty()) {
            end = "end_mask=\"";
            for (const auto& pb : p.end_mask)
                end += std::to_string(pb) + ",";
            end.pop_back();
            end += "\"";
        }
        REPLACE_WITH_STR(model, "_END_", end);

        if (!p.ellipsis_mask.empty()) {
            ellipsis = "ellipsis_mask=\"";
            for (const auto& pb : p.ellipsis_mask)
                ellipsis += std::to_string(pb) + ",";
            ellipsis.pop_back();
            ellipsis += "\"";
        }
        REPLACE_WITH_STR(model, "_ELLIPSIS_", ellipsis);

        if (!p.new_axis_mask.empty()) {
            new_axis = "new_axis_mask=\"";
            for (const auto& pb : p.new_axis_mask)
                new_axis += std::to_string(pb) + ",";
            new_axis.pop_back();
            new_axis += "\"";
        }
        REPLACE_WITH_STR(model, "_NEW_AXIS_", new_axis);

        if (!p.shrink_axis_mask.empty()) {
            shrink_axis = "shrink_axis_mask=\"";
            for (const auto& pb : p.shrink_axis_mask)
                shrink_axis += std::to_string(pb) + ",";
            shrink_axis.pop_back();
            shrink_axis += "\"";
        }
        REPLACE_WITH_STR(model, "_SHRINK_", shrink_axis);

        for (const auto& i : p.out_shape) {
            out_shape += "<dim>";
            out_shape += std::to_string(i) + "</dim>\n";
        }
        out_shape.pop_back();
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        return model;
    }

    static InferenceEngine::TBlob<uint8_t>::Ptr generateWeights(const std::vector<std::vector<int32_t>> &data) {
        size_t totalSize = 0;
        for (const auto & i : data)
            totalSize += i.size();
        auto weights = new InferenceEngine::TBlob<uint8_t>(
            {InferenceEngine::Precision::U8, { totalSize * sizeof(int32_t) }, InferenceEngine::C});
        weights->allocate();
        size_t vectorCounter = 0;
        size_t innerVectorCounter = 0;
        for (size_t i = 0; i < totalSize; i++) {
            if (innerVectorCounter >= data[vectorCounter].size()) {
                ++vectorCounter;
                innerVectorCounter = 0;
            }
            weights->data().as<int32_t*>()[i] = data[vectorCounter][innerVectorCounter];
            ++innerVectorCounter;
        }
        return InferenceEngine::TBlob<uint8_t>::Ptr(weights);
    }
};

TEST_P(myriadLayersTestsStridedSlice_smoke, TestsStridedSlice) {
    auto p = ::testing::WithParamInterface<strided_slice_test_param>::GetParam();

    std::string model = getModel(p);

    TBlob<uint8_t>::Ptr weights(generateWeights({ p.begin, p.end, p.strides }));
    // Parse model.
    ASSERT_NO_THROW(readNetwork(model, weights));

    const auto& network = _cnnNetwork;

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setPrecision(Precision::FP16);
    auto outputsInfo = network.getOutputsInfo();
    outputsInfo["strided_slice"]->setPrecision(Precision::FP16);

    // Load network.

    std::map<std::string, std::string> config = {
        { InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO) }
    };
    if (!CheckMyriadX()) {
        config.insert({ InferenceEngine::MYRIAD_DISABLE_REORDER, CONFIG_VALUE(YES) });
    }

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, config));

    // Create InferRequest.
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = _exeNetwork.CreateInferRequest());

    // Input Data.
    InferenceEngine::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob("input"));
    GenRandomData(inputBlob);

    // Infer & get output blob.
    InferenceEngine::Blob::Ptr outputBlob;
    ASSERT_NO_THROW(inferRequest.Infer());
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob("strided_slice"));

    // Output Reference.
    Blob::Ptr refBlob = InferenceEngine::make_shared_blob<ie_fp16>(outputBlob->getTensorDesc());
    refBlob->allocate();

    // Check results.
    InferenceEngine::SizeVector out_dims;
    ref_strided_slice(inputBlob, refBlob, out_dims, p.begin, p.end, p.strides, p.begin_mask, p.end_mask);

    // Check out shapes.
    if(out_dims.size() != p.out_shape.size())
        FAIL() << "Wrong out_shape size!";
    for (size_t i = 0; i < p.out_shape.size(); i++) {
        if (out_dims[i] != p.out_shape[i])
            FAIL() << "Wrong out_shape dimensions!";
    }

    CompareCommonAbsolute(outputBlob, refBlob, 0);
}

// Params: in_shape, dim_size, begin, end, stride, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out_shape
static std::vector<strided_slice_test_param> s_stridedSliceParams = {
    strided_slice_test_param{ { 10 }, 1, { 0 }, { 10 }, { 2 }, { 1 }, { 1 }, {}, {}, {}, { 5 } },
    strided_slice_test_param{ { 10 }, 1, { 1 }, { 9 }, { 2 }, { 1 }, { 1 }, {}, {}, {}, { 4 } },
    strided_slice_test_param{ { 10 }, 1, { 1 }, { 9 }, { 2 }, { 0 }, { 1 }, {}, {}, {}, { 5 } },
    strided_slice_test_param{ { 1000, 4 }, 2, { 0, 0 }, { 1000, 4 }, { 1, 4 }, { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 1000, 1 } },
    strided_slice_test_param{ { 1000, 4 }, 2, { 200, 1 }, { 500, 3 }, { 1, 2 }, { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 1000, 1 } },
    strided_slice_test_param{ { 1, 2, 35, 33 }, 4, { 0, 0, 0, 2 }, { 1, 2, 33, 31 }, {1, 1, 1, 2}, { 0, 0, 0, 1 }, { 0, 0, 1, 1 }, {}, {}, {}, { 1, 2, 33, 15 } },
    strided_slice_test_param{ { 2, 2, 2, 3}, 4, { 0, 0, 0, 1 }, { 2, 2, 2, 3 }, { 1, 2, 2, 2 }, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, {}, {}, {}, { 2, 1, 1, 1 } },
    strided_slice_test_param{ { 2, 8, 32, 32}, 4, { 0, 2, 0, 0 }, { 2, 7, 0, 0 }, { 1, 3, 1, 1 }, { 0, 1, 0, 0 }, { 0, 1, 0, 0 }, {}, {}, {}, { 2, 2, 32, 32 } },
    strided_slice_test_param{ { 2, 8, 32, 32}, 4, { 0, 0, 2, 0 }, { 0, 0, 31, 0 }, { 1, 1, 3, 1 }, { 0, 0, 1, 0 }, { 0, 0, 1, 0 }, {}, {}, {}, { 2, 8, 10, 32 } },
    strided_slice_test_param{ { 2, 8, 32, 32}, 4, { 0, 0, 0, 2 }, { 0, 0, 0, 0 }, { 1, 1, 1, 3 }, { 0, 0, 0, 1 }, { 0, 0, 0, 0 }, {}, {}, {}, { 2, 8, 32, 10 } },
    strided_slice_test_param{ { 1, 32, 128, 128 }, 4, {0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 1, 2, 4, 8 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, {}, {}, {}, { 1, 16, 32, 16 } },
    strided_slice_test_param{ { 1, 32, 128, 128 }, 4, {0, 16, 0, 0 }, { 0, 0, 0, 0 }, {}, { 0, 1, 0, 0 }, { 0, 0, 0, 0 }, {}, {}, {}, { 1, 16, 128, 128 } },
    strided_slice_test_param{ { 4, 1000 }, 2, { 0, 0 }, { 4, 9999 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 4, 1000 } },
    strided_slice_test_param{ { 4, 1000 }, 2, { 0, 0 }, { 4, -1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 4, 999 } },
    strided_slice_test_param{ { 4, 1000 }, 2, { 0, 0 }, { 4, -3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 4, 997 } },
};
