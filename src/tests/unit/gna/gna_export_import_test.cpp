// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <ie_core.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gna_mock_api.hpp"
#include "gna_plugin.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using GNAPluginNS::GNAPlugin;
using namespace InferenceEngine;

class GNAExportImportTest : public ::testing::Test {
public:
    void ExportModel(std::string exportedModelFileName) {
        auto function = getFunction();
        auto weights = make_shared_blob<uint8_t>({ Precision::U8, {1, 10}, Layout::NC });
        weights->allocate();
        fillWeights(weights);
        Core core;
        CNNNetwork cnnNetwork = CNNNetwork{function};

        GNACppApi mockApi;
        std::vector<std::vector<uint8_t>> data;
        ExpectEnqueueCalls(&mockApi, data);
        GNAPlugin plugin(gna_config);

        plugin.LoadNetwork(cnnNetwork);
        plugin.Export(exportedModelFileName);
    }

    void ImportModel(std::string modelPath) {
        GNACppApi mockApi;
        std::vector<std::vector<uint8_t>> data;
        ExpectEnqueueCalls(&mockApi, data);
        GNAPlugin plugin(gna_config);
        std::fstream inputStream(modelPath, std::ios_base::in | std::ios_base::binary);
        if (inputStream.fail()) {
            THROW_GNA_EXCEPTION << "Cannot open file to import model: " << modelPath;
        }

        auto sp = plugin.ImportNetwork(inputStream);
        auto inputsInfo = plugin.GetNetworkInputs();
        auto outputsInfo = plugin.GetNetworkOutputs();

        BlobMap input, output;
        AllocateInput(input, &plugin);
        AllocateOutput(output, &plugin);
        plugin.Infer(input, output);
    }

protected:
    void AllocateInput(BlobMap& input, GNAPlugin *plugin) {
        auto inputsInfo = plugin->GetNetworkInputs();
        for (auto&& info : inputsInfo) {
            std::map<std::string, std::vector<float>>::iterator it;
            auto & inputBlob = input[info.first];
            inputBlob = make_blob_with_precision({ Precision::FP32, info.second->getTensorDesc().getDims(),
                info.second->getLayout() });
            inputBlob->allocate();
        }
    }

    void AllocateOutput(BlobMap& output, GNAPlugin *plugin) {
        auto outputsInfo = plugin->GetNetworkOutputs();
        for (auto&& out : outputsInfo) {
            auto& outputBlob = output[out.first];
            auto dims = out.second->getDims();
            auto outsize = details::product(std::begin(dims), std::end(dims));
            outputBlob.reset(new TBlob<float>({ Precision::FP32, {1, outsize}, Layout::NC }));
            outputBlob->allocate();
        }
    }

    void fillWeights(InferenceEngine::Blob::Ptr weights, std::vector<float> pattern = {(1.0F)}) {
        float * p = weights->buffer().as<float *>();
        float * pEnd = p + weights->byteSize() / sizeof(float);

        for (; p != pEnd ;) {
            for (int i = 0; i != (weights->byteSize() / sizeof(float) / 3) + 1; i++) {
                for (int j = 0; j != pattern.size() && p != pEnd; j++, p++) {
                    *p = pattern[j];
                }
            }
        }
    }

    std::shared_ptr<ngraph::Function> getFunction() {
        auto ngPrc = ngraph::element::f32;
        size_t shape = 10;
        auto params = ngraph::builder::makeParams(ngPrc, {{1, shape}});
        auto mul_const = ngraph::builder::makeConstant<float>(ngPrc, { shape, shape },
            CommonTestUtils::generate_float_numbers(shape * shape, -0.5f, 0.5f), false);

        auto matmul = std::make_shared<ngraph::op::MatMul>(params[0], mul_const, false, true);
        auto res = std::make_shared<ngraph::op::Result>(matmul);
        auto function = std::make_shared<ngraph::Function>(res, params, "MatMul");
        return function;
    }

    void ExpectEnqueueCalls(GNACppApi *mockApi, std::vector<std::vector<uint8_t>>& data) {
        EXPECT_CALL(*mockApi, Gna2MemoryAlloc(_, _, _)).Times(AtLeast(1)).WillRepeatedly(Invoke([&data](
            uint32_t sizeRequested,
            uint32_t *sizeGranted,
            void **memoryAddress) {
                data.push_back(std::vector<uint8_t>(sizeRequested));
                *sizeGranted = sizeRequested;
                *memoryAddress = data.back().data();
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*mockApi, Gna2DeviceGetVersion(_, _)).WillOnce(Invoke([](
            uint32_t deviceIndex,
            enum Gna2DeviceVersion * deviceVersion) {
                *deviceVersion = Gna2DeviceVersionSoftwareEmulation;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*mockApi, Gna2DeviceOpen(_)).WillOnce(Return(Gna2StatusSuccess));

        EXPECT_CALL(*mockApi, Gna2GetLibraryVersion(_, _)).Times(AtLeast(0)).WillRepeatedly(Return(Gna2StatusSuccess));

        EXPECT_CALL(*mockApi, Gna2InstrumentationConfigCreate(_, _, _, _)).WillOnce(Return(Gna2StatusSuccess));

        EXPECT_CALL(*mockApi, Gna2ModelCreate(_, _, _)).WillOnce(Invoke([](
            uint32_t deviceIndex,
            struct Gna2Model const * model,
            uint32_t * modelId) {
                *modelId = 0;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*mockApi, Gna2RequestConfigCreate(_, _)).WillOnce(Invoke([](
            uint32_t modelId,
            uint32_t * requestConfigId) {
                *requestConfigId = 0;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*mockApi, Gna2InstrumentationConfigAssignToRequestConfig(_, _)).Times(AtLeast(1)).WillRepeatedly(Return(Gna2StatusSuccess));
    }
    void TearDown() override {
        std::remove(exported_file_name.c_str());
    }
    std::map<std::string, std::string> gna_config;
    std::string exported_file_name;
};

TEST_F(GNAExportImportTest, ExportImportI16) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_PRECISION", "I16"}
    };
    exported_file_name = "export_test.bin";
    ExportModel(exported_file_name);
    ImportModel(exported_file_name);
}

TEST_F(GNAExportImportTest, ExportImportI8) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_PRECISION", "I8"}
    };
    exported_file_name = "export_test.bin";
    ExportModel(exported_file_name);
    ImportModel(exported_file_name);
}
