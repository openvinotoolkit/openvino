// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"

#include <debug.h>

#include <algorithm>
#include <functional>
#include <string>

using namespace InferenceEngine;

typedef struct {
    SizeVector dims;
    int axis;
    int k;
} Geometry;

void PrintTo(const Geometry& p, std::ostream* os) {
    *os << "{dims:" << details::dumpVec(p.dims) << ", axis:" << p.axis << ", k:" << p.k << "}";
}

using TopKTestParams = std::tuple<Geometry, std::string, std::string>;

static const Precision dataPrecision = Precision::FP16;
static const Precision indexPrecision = Precision::I32;

class TopKTest: public myriadLayerTestBaseWithParam<TopKTestParams>
{
protected:
    std::set<std::string> getExecutedStagesTypes() const {
        std::set<std::string> result;
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        _inferRequest->GetPerformanceCounts(perfMap, nullptr);

        for (const auto& perf : perfMap)
            result.emplace(perf.second.exec_type);

        return result;
    }

    void testTopK(const IRVersion irVersion, const bool outputValues, const bool outputIndices) {
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
        _config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
        _irVersion = irVersion;

        // Skipping outputs available only for v10.
        ASSERT_TRUE(irVersion == IRVersion::v10 || outputValues && outputIndices);

        const auto params = GetParam();
        const auto geometry = std::get<0>(params);
        const auto inputDims = geometry.dims;
        const auto axis = geometry.axis;
        const auto k = geometry.k;
        const auto mode = std::get<1>(params);
        const auto sort = std::get<2>(params);

        const auto outputDims = calcOutputDims(inputDims, axis, k);
        const auto model = irVersion == IRVersion::v10
                ? getModelV10(inputDims, outputDims, axis, mode, sort, outputValues, outputIndices)
                : getModelV7(inputDims, outputDims, axis, mode, sort);

        TBlob<uint8_t>::Ptr weightsBlob;
        TBlob<int32_t>::Ptr inputKBlob;
        getKBlob(k, weightsBlob, inputKBlob);
        ASSERT_NE(weightsBlob, nullptr);

        ASSERT_NO_THROW(readNetwork(model, weightsBlob));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["topk_input"]->setPrecision(dataPrecision);
        _inputsInfo["topk_input"]->setLayout(defaultLayout(inputDims.size()));

        _outputsInfo = network.getOutputsInfo();
        if (outputValues) {
            _outputsInfo["topk.0"]->setPrecision(dataPrecision);
            _outputsInfo["topk.0"]->setLayout(defaultLayout(outputDims.size()));
        }
        if (outputIndices) {
            _outputsInfo["topk.1"]->setPrecision(indexPrecision);
            _outputsInfo["topk.1"]->setLayout(defaultLayout(outputDims.size()));
        }

        StatusCode st = OK;

        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, _config, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr inputValuesBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("topk_input", inputValuesBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        GenRandomData(inputValuesBlob);

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        const auto executedTypes = getExecutedStagesTypes();

        // This logic must be synchronized with TopKStage class.
        const bool useArgMaxOptimization = (!outputValues || !outputIndices)
                && mode == "max"
                && ((sort == "value" && outputValues) || (sort == "index" && outputIndices));

        ASSERT_EQ(executedTypes.count("ArgMax"), useArgMaxOptimization);
        ASSERT_EQ(executedTypes.count("TopK"), !useArgMaxOptimization);

        Blob::Ptr outputValuesBlob, outputIndicesBlob;
        if (outputValues) {
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("topk.0", outputValuesBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        }
        if (outputIndices) {
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("topk.1", outputIndicesBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        }

        const InferenceEngine::TensorDesc valuesDesc{dataPrecision, outputDims, defaultLayout(outputDims.size())};
        const InferenceEngine::TensorDesc indicesDesc{indexPrecision, outputDims, defaultLayout(outputDims.size())};

        Blob::Ptr refValuesBlob = make_shared_blob<ie_fp16>(valuesDesc);
        refValuesBlob->allocate();
        Blob::Ptr refIndicesBlob = make_shared_blob<int32_t>(indicesDesc);
        refIndicesBlob->allocate();

        ref_topk(inputValuesBlob, inputKBlob, refValuesBlob, refIndicesBlob, axis, mode, sort);
        if (outputValues)
            CompareCommonAbsolute(outputValuesBlob, /*expected=*/refValuesBlob, 0.0f);

        if (outputIndices)
            CompareCommonExact(outputIndicesBlob, /*expected=*/refIndicesBlob);
    }

    static std::string getModelV7(const SizeVector& inputDims,
                                  const SizeVector& outputDims, int axis,
                                  const std::string& mode, const std::string& sort) {
        std::string model = R"V0G0N(
            <net name="testTopK" version="7">
                <layers>
                    <layer id="0" name="topk_input" type="Input">
                        <output>
                            <port id="0" precision="__DATA_PRECISION__">__INPUT_DIMS__</port>
                        </output>
                    </layer>
                    <layer id="1" name="topk_k" type="Const">
                        <output>
                            <port id="1" precision="__INDEX_PRECISION__">__K_DIMS__</port>
                        </output>
                        <blobs>
                            <custom offset="0" size="__K_SIZE__"/>
                        </blobs>
                    </layer>
                    <layer id="2" name="topk" type="TopK">
                        <data axis="__AXIS__" mode="__MODE__" sort="__SORT__"/>
                        <input>
                            <port id="0">__INPUT_DIMS__</port>
                            <port id="1">__K_DIMS__</port>
                        </input>
                        <output>
                            <port id="2" precision="__DATA_PRECISION__">__OUTPUT_DIMS__</port>
                            <port id="3" precision="__INDEX_PRECISION__">__OUTPUT_DIMS__</port>
                        </output>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                    <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                </edges>
            </net>
        )V0G0N";

        const std::string inputDimsStr = dimsToString(inputDims);
        const std::string kDims = dimsToString({1});
        const std::string outputDimsStr = dimsToString(outputDims);
        const size_t kSize = sizeof(int32_t);

        REPLACE_WITH_STR(model, "__DATA_PRECISION__", dataPrecision.name());
        REPLACE_WITH_STR(model, "__INDEX_PRECISION__", indexPrecision.name());
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", inputDimsStr);
        REPLACE_WITH_STR(model, "__K_DIMS__", kDims);
        REPLACE_WITH_NUM(model, "__K_SIZE__", kSize);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", outputDimsStr);
        REPLACE_WITH_NUM(model, "__AXIS__", axis);
        REPLACE_WITH_STR(model, "__MODE__", mode);
        REPLACE_WITH_STR(model, "__SORT__", sort);

        return model;
    }
    static std::string getModelV10(const SizeVector& inputDims,
                                const SizeVector& outputDims, int axis,
                                const std::string& mode, const std::string& sort,
                                const bool outputValues, const bool outputIndices) {
        std::string model = R"V0G0N(
            <net name="testTopK" version="10">
                <layers>
                    <layer id="0" name="topk_input" type="Parameter" version="opset1">
                        <data element_type="f16" shape="__INPUT_DIMS_SHAPE__"/>
                        <output>
                            <port id="0" precision="__DATA_PRECISION__">__INPUT_DIMS__</port>
                        </output>
                    </layer>
                    <layer id="1" name="topk_k" type="Const" version="opset1">
                        <data element_type="f16" offset="0" shape="__K_DIMS_SHAPE__" size="__K_SIZE__"/>
                        <output>
                            <port id="1" precision="__INDEX_PRECISION__" />
                        </output>
                    </layer>
                    <layer id="2" name="topk" type="TopK" version="opset1">
                        <data axis="__AXIS__" mode="__MODE__" sort="__SORT__"/>
                        <input>
                            <port id="0">__INPUT_DIMS__</port>
                            <port id="1" />
                        </input>
                        <output>
                            <port id="2" precision="__DATA_PRECISION__">__OUTPUT_DIMS__</port>
                            <port id="3" precision="__INDEX_PRECISION__">__OUTPUT_DIMS__</port>
                        </output>
                    </layer>
                    __RESULT_LAYERS__
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                    <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                    __RESULT_EDGES__
                </edges>
            </net>
        )V0G0N";

        const std::string inputDimsStr  = dimsToString(inputDims);
        const std::string outputDimsStr = dimsToString(outputDims);

        const size_t kSize = sizeof(int32_t);

        /// TODO: consider extending IRDumperNetwork to support this with OOP API.
        /// At the moment layers with multiple outputs not supported
        std::string resultLayers, resultEdges;
        auto addResultLayer = [&resultLayers, &resultEdges, &outputDimsStr]
                (const std::string& name, const std::string& id, const std::string& sourcePort){

            std::string result = R"V0G0N(
               <layer id="__ID__" name="__NAME__" type="Result" version="opset1">
                   <input>
                       <port id="0">__OUTPUT_DIMS__</port>
                   </input>
               </layer>
               )V0G0N";
             REPLACE_WITH_STR(result, "__ID__", id);
             REPLACE_WITH_STR(result, "__NAME__", name);
             REPLACE_WITH_STR(result, "__OUTPUT_DIMS__", outputDimsStr);
             resultLayers += result;
             resultEdges  += "<edge from-layer=\"2\" from-port=\"" + sourcePort + "\" to-layer=\"" + id + "\" to-port=\"0\"/>";
        };

        if (outputValues)
            addResultLayer("topk.0", "3", "2");

        if (outputIndices)
            addResultLayer("topk.1", "4", "3");


        REPLACE_WITH_STR(model, "__DATA_PRECISION__", dataPrecision.name());
        REPLACE_WITH_STR(model, "__INDEX_PRECISION__", indexPrecision.name());
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", inputDimsStr);
        REPLACE_WITH_NUM_VECTOR(model, "__INPUT_DIMS_SHAPE__", inputDims);
        REPLACE_WITH_STR(model, "__K_DIMS_SHAPE__", "1");
        REPLACE_WITH_NUM(model, "__K_SIZE__", kSize);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", outputDimsStr);
        REPLACE_WITH_NUM(model, "__AXIS__", axis);
        REPLACE_WITH_STR(model, "__MODE__", mode);
        REPLACE_WITH_STR(model, "__SORT__", sort);
        REPLACE_WITH_STR(model, "__RESULT_LAYERS__", resultLayers);
        REPLACE_WITH_STR(model, "__RESULT_EDGES__", resultEdges);

        return model;
    }

    static std::string dimsToString(const SizeVector& dims) {
        std::string str;
        for (auto& d : dims)
            str += "<dim>" + std::to_string(d) + "</dim>";
        return str;
    }

    static SizeVector calcOutputDims(const SizeVector& inputDims, int axis, int k) {
        SizeVector outputDims = inputDims;
        outputDims[axis] = k;
        return outputDims;
    }
    static Layout defaultLayout(int ndims) {
        switch (ndims) {
        case 5: return NCDHW;
        case 4: return NCHW;
        case 3: return CHW;
        case 2: return NC;
        case 1: return C;
        }
        return ANY;
    }
    static void getKBlob(int k, TBlob<uint8_t>::Ptr& weightsBlob, TBlob<int32_t>::Ptr& kBlob) {
        const size_t k_size = 1;
        const size_t weights_size = k_size * sizeof(int32_t);

        TBlob<uint8_t>* weights_raw = new TBlob<uint8_t>(TensorDesc(Precision::U8, {weights_size}, C));
        weights_raw->allocate();
        int32_t* weightsData = weights_raw->data().as<int32_t*>();

        TBlob<int32_t>* k_raw = new TBlob<int32_t>(TensorDesc(Precision::I32, {k_size}, C));
        k_raw->allocate();
        int32_t* kData = k_raw->data().as<int32_t*>();

        weightsData[0] = k;
        kData[0] = k;

        weightsBlob = TBlob<uint8_t>::Ptr(weights_raw);
        kBlob = TBlob<int32_t>::Ptr(k_raw);
    }
};

class myriadTestsTopK_nightly: public TopKTest
{
};

TEST_P(myriadTestsTopK_nightly, TopKv7)
{
    testTopK(IRVersion::v7, true, true);
}

TEST_P(myriadTestsTopK_nightly, TopKv10_All)
{
    testTopK(IRVersion::v10, true, true);
}

TEST_P(myriadTestsTopK_nightly, TopKv10_ArgMaxValues)
{
    testTopK(IRVersion::v10, true, false);
}

TEST_P(myriadTestsTopK_nightly, TopKv10_ArgMaxIndices)
{
    testTopK(IRVersion::v10, false, true);
}
