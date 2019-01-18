// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <extension/ext_list.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct gather_test_params {
    std::string inIdxPrecision;
    InferenceEngine::SizeVector inIdx;
    InferenceEngine::SizeVector inDict;
    int axis;
    InferenceEngine::SizeVector out;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};


inline void clipping(int *idx, const int min, const int max) {
    (*idx) = ((*idx) > min) ? (*idx) : min;
    (*idx) = ((*idx) < max) ? (*idx) : (max - 1);
    return;
}

template <typename data_t>
void ref_gather(InferenceEngine::TBlob<data_t> &srcIdx, InferenceEngine::TBlob<float> &srcDct, InferenceEngine::TBlob<float> &dst, size_t axis) {
    size_t i, j;
    const data_t *src_dataIdx = srcIdx.data();
    float* src_dataDict = srcDct.data();
    float *dst_data = dst.data();
    size_t src_size = srcIdx.size();

    std::vector<size_t> dims = srcDct.getTensorDesc().getDims();
    std::vector<size_t> dims_actual;

    //  Remove redundant dimensions
    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] > 1) {
            for (size_t j = i; j < dims.size(); j++)
                dims_actual.push_back(dims[j]);
            break;
        }
    }

    //  Find number of dictionaries, index range and data length
    size_t numDictionaries = 1;
    for (i = 0; i < axis; i++)
        numDictionaries *= dims_actual[i];
    size_t indexRange = dims_actual[axis];
    size_t dataLength = 1;
    for (i = axis + 1; i < dims_actual.size(); i++)
        dataLength *= dims_actual[i];

    //  The gathering process
    for (i = 0; i < src_size; i++) {
        int idx = static_cast<int>(src_dataIdx[i]);

        //  Index clipping
        clipping(&idx, 0, indexRange);

        //  Copying data to destination from Dictionary
        for (j = 0; j < numDictionaries; j++) {
            memcpy(&dst_data[dataLength * (i + j * src_size)],
                   &src_dataDict[dataLength * (idx + j * indexRange)], sizeof(float)*dataLength);
        }
    }
}

class MKLDNNCPUExtGatherTests: public TestsCommon, public WithParamInterface<gather_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Gather_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputText" type="Input" precision="_IIDXP_" id="1">
            <output>
                <port id="1">
                    _IIDX_
                </port>
            </output>
        </layer>
        <layer name="InputDictionary" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    _IDICT_
                </port>
            </output>
        </layer>
        <layer name="gather" id="3" type="Gather" precision="FP32">
            <data axis="_AX_"/>
            <input>
                <port id="1">
                    _IDICT_
                </port>
                <port id="2">
                    _IIDX_
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(gather_test_params p) {
        std::string model = model_t;
        std::string inIdx;
        std::string inDict;
        std::string out;

        for (auto& idx : p.inIdx) {
            inIdx += "<dim>";
            inIdx += std::to_string(idx) + "</dim>\n";
        }

        for (auto& dct : p.inDict) {
            inDict += "<dim>";
            inDict += std::to_string(dct) + "</dim>\n";
        }

        for (auto& dst : p.out) {
            out += "<dim>";
            out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IIDXP_", p.inIdxPrecision);
        REPLACE_WITH_STR(model, "_IIDX_", inIdx);
        REPLACE_WITH_STR(model, "_IDICT_", inDict);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_STR(model, "_OUT_", out);

        return model;
    }

    template <typename data_t>
    static void fill_data_dbgval(data_t *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<data_t>(i & (sizeof(data_t) * 8 - 1));
        }
    }
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gather_test_params p = ::testing::WithParamInterface<gather_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "gather") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            ASSERT_EQ(4, nodes.size());

            // Input Dictionary
            InferenceEngine::Blob::Ptr srcDict = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.inDict, InferenceEngine::TensorDesc::getLayoutByDims(p.inDict) });
            srcDict->allocate();
            fill_data(srcDict->buffer(), srcDict->size());
            auto * srcDictPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcDict.get());
            if (srcDictPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            // Input Indexes
            InferenceEngine::Blob::Ptr srcIdx;
            if (p.inIdxPrecision == "I32") {
                srcIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, p.inIdx, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdx) });
                srcIdx->allocate();
                fill_data_dbgval(static_cast<int32_t*>(srcIdx->buffer()), srcIdx->size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                // Check results
                ref_gather(*srcIdxPtr, *srcDictPtr, dst_ref, p.axis);
            }
            else if (p.inIdxPrecision == "FP32") {
                srcIdx = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.inIdx, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdx) });
                srcIdx->allocate();
                fill_data(srcIdx->buffer(), srcIdx->size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                // Check results
                ref_gather(*srcIdxPtr, *srcDictPtr, dst_ref, p.axis);
            }
            else if (p.inIdxPrecision == "U16") {
                srcIdx = InferenceEngine::make_shared_blob<uint16_t>({ InferenceEngine::Precision::U16, p.inIdx, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdx) });
                srcIdx->allocate();
                fill_data_dbgval(static_cast<uint16_t*>(srcIdx->buffer()), srcIdx->size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<uint16_t>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<uint16_t>.";

                // Check results
                ref_gather(*srcIdxPtr, *srcDictPtr, dst_ref, p.axis);
            }
            else if (p.inIdxPrecision == "I16") {
                srcIdx = InferenceEngine::make_shared_blob<int16_t>({ InferenceEngine::Precision::I16, p.inIdx, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdx) });
                srcIdx->allocate();
                fill_data_dbgval(static_cast<int16_t*>(srcIdx->buffer()), srcIdx->size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<int16_t>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int16_t>.";

                // Check results
                ref_gather(*srcIdxPtr, *srcDictPtr, dst_ref, p.axis);
            }
            else if (p.inIdxPrecision == "U8") {
                srcIdx = InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, p.inIdx, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdx) });
                srcIdx->allocate();
                fill_data_dbgval(static_cast<uint8_t*>(srcIdx->buffer()), srcIdx->size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<uint8_t>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<uint8_t>.";

                // Check results
                ref_gather(*srcIdxPtr, *srcDictPtr, dst_ref, p.axis);
            }
            else if (p.inIdxPrecision == "I8") {
                srcIdx = InferenceEngine::make_shared_blob<int8_t>({ InferenceEngine::Precision::I8, p.inIdx, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdx) });
                srcIdx->allocate();
                fill_data_dbgval(static_cast<int8_t*>(srcIdx->buffer()), srcIdx->size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<int8_t>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int8_t>.";

                // Check results
                ref_gather(*srcIdxPtr, *srcDictPtr, dst_ref, p.axis);
            }
            else {
                return;
            }

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputDictionary", srcDict));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputText", srcIdx));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtGatherTests, TestsGather) {}

INSTANTIATE_TEST_CASE_P(
        TestsGather, MKLDNNCPUExtGatherTests,
            ::testing::Values(
                gather_test_params{ "FP32", {1, 1, 12, 256}, {1, 1, 71, 16}, 0, {1, 12, 256, 16}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {1, 1, 12, 256}, {1, 1, 71, 16}, 0, {1, 12, 256, 16}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I16", {1, 1, 12, 256}, {1, 1, 71, 16}, 0, {1, 12, 256, 16}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{   "U8", {1, 1, 12, 256}, {1, 1, 71, 16}, 0, {1, 12, 256, 16}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{   "I8", {1, 1, 12, 256}, {1, 1, 71, 16}, 0, {1, 12, 256, 16}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {12, 256}, {71, 16}, 0, {12, 256, 16}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {3, 4}, {2, 5, 6}, 0, {3, 4, 5, 6}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {3, 4}, {5, 1}, 0, {3, 4, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{ "FP32", {1, 1, 12, 256}, {1, 1, 71, 16}, 1, {1, 71, 12, 256}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {1, 1, 3, 4}, {1, 2, 5, 6}, 1, {2, 3, 4, 6}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {1, 1, 3, 4}, {1, 2, 5, 6}, 2, {2, 5, 3, 4}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                gather_test_params{  "I32", {12, 4, 9, 8}, {6, 13, 10, 3}, 1, {6, 12, 4, 9, 8, 10, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown }
            ));




struct gatherTF_test_params {
    InferenceEngine::SizeVector in_dim;
    std::vector<int32_t> in;

    InferenceEngine::SizeVector dct_dim;
    std::vector<float> dct;

    int axis;

    InferenceEngine::SizeVector ref_dim;
    std::vector<float> ref;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtGatherTFTests : public TestsCommon, public WithParamInterface<gatherTF_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Gather_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputText" type="Input" precision="I32" id="1">
            <output>
                <port id="1">
                    _IIDX_
                </port>
            </output>
        </layer>
        <layer name="InputDictionary" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    _IDICT_
                </port>
            </output>
        </layer>
        <layer name="gather" id="3" type="Gather" precision="FP32">
            <data axis="_AX_"/>
            <input>
                <port id="1">
                    _IDICT_
                </port>
                <port id="2">
                    _IIDX_
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(gatherTF_test_params p) {
        std::string model = model_t;
        std::string inIdx;
        std::string inDict;
        std::string out;

        for (auto& idx : p.in_dim) {
            inIdx += "<dim>";
            inIdx += std::to_string(idx) + "</dim>\n";
        }

        for (auto& dct : p.dct_dim) {
            inDict += "<dim>";
            inDict += std::to_string(dct) + "</dim>\n";
        }

        for (auto& dst : p.ref_dim) {
            out += "<dim>";
            out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IIDX_", inIdx);
        REPLACE_WITH_STR(model, "_IDICT_", inDict);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_STR(model, "_OUT_", out);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gatherTF_test_params p = ::testing::WithParamInterface<gatherTF_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            // Input Indexes
            InferenceEngine::Blob::Ptr srcIdx;
            srcIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, p.in_dim, InferenceEngine::TensorDesc::getLayoutByDims(p.in_dim) });
            srcIdx->allocate();
            memcpy(static_cast<int32_t*>(srcIdx->buffer()), &p.in[0], sizeof(int32_t)*p.in.size());
            auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(srcIdx.get());
            if (srcIdxPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<int32_t>.";

            //  Input Dictionary
            InferenceEngine::Blob::Ptr srcDict = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.dct_dim, InferenceEngine::TensorDesc::getLayoutByDims(p.dct_dim) });
            srcDict->allocate();
            memcpy(srcDict->buffer(), &p.dct[0], sizeof(float)*p.dct.size());
            auto * srcDictPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcDict.get());
            if (srcDictPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            //  Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            //  Infer
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputDictionary", srcDict));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputText", srcIdx));
            graph.Infer(srcs, outputBlobs);

            //  Check results
            if (memcmp((*output).data(), &p.ref[0], p.ref.size()) != 0)
                FAIL() << "Wrong result with compare TF reference!";
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtGatherTFTests, TestsGather) {}

//  Test data vectors
std::vector<int32_t> in0 = { 0, 1, 1, 0 };
std::vector<int32_t> in1 = { 0, 1, 2, 1 };
std::vector<float> dict = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f };
std::vector<float> ref_in0_a0_d223 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }; // 2x2x2x3
std::vector<float> ref_in0_a2_d232 = { 1.f, 2.f, 2.f, 1.f, 3.f, 4.f, 4.f, 3.f, 5.f, 6.f, 6.f, 5.f, 7.f, 8.f, 8.f, 7.f, 9.f, 10.f, 10.f, 9.f, 11.f, 12.f, 12.f, 11.f }; // 2x3x2x2
std::vector<float> ref_in1_a0_d322 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f }; // 2x2x2x2
std::vector<float> ref_in1_a1_d232 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f }; // 2x2x2x2
std::vector<float> ref_in1_a2_d223 = { 1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f }; // 2x2x2x2

INSTANTIATE_TEST_CASE_P(
        TestsGather, MKLDNNCPUExtGatherTFTests,
        ::testing::Values(
        gatherTF_test_params{ { 2, 2 }, in0,{ 2, 2, 3 }, dict, 0, { 2, 2, 2, 3 }, ref_in0_a0_d223 },
        gatherTF_test_params{ { 2, 2 }, in0,{ 2, 2, 3 }, dict,-3, { 2, 2, 2, 3 }, ref_in0_a0_d223 },
        gatherTF_test_params{ { 2, 2 }, in0,{ 2, 3, 2 }, dict, 2, { 2, 3, 2, 2 }, ref_in0_a2_d232 },
        gatherTF_test_params{ { 2, 2 }, in0,{ 2, 3, 2 }, dict,-1, { 2, 3, 2, 2 }, ref_in0_a2_d232 },
        gatherTF_test_params{ { 2, 2 }, in1,{ 3, 2, 2 }, dict, 0, { 2, 2, 2, 2 }, ref_in1_a0_d322 },
        gatherTF_test_params{ { 2, 2 }, in1,{ 3, 2, 2 }, dict,-3, { 2, 2, 2, 2 }, ref_in1_a0_d322 },
        gatherTF_test_params{ { 2, 2 }, in1,{ 2, 3, 2 }, dict, 1, { 2, 2, 2, 2 }, ref_in1_a1_d232 },
        gatherTF_test_params{ { 2, 2 }, in1,{ 2, 3, 2 }, dict,-2, { 2, 2, 2, 2 }, ref_in1_a1_d232 },
        gatherTF_test_params{ { 2, 2 }, in1,{ 2, 2, 3 }, dict, 2, { 2, 2, 2, 2 }, ref_in1_a2_d223 },
        gatherTF_test_params{ { 2, 2 }, in1,{ 2, 2, 3 }, dict,-1, { 2, 2, 2, 2 }, ref_in1_a2_d223 }));


class MKLDNNCPUExtGatherHolesTests : public TestsCommon, public WithParamInterface<gatherTF_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Gather_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputText" type="Input" precision="I32" id="1">
            <output>
                <port id="1">
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="InputDictionary" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="Input3" type="Input" precision="FP32" id="3">
            <output>
                <port id="3">
                    <dim>2</dim>
                    <dim>5</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="gather" id="4" type="Gather" precision="FP32">
            <data axis="0"/>
            <input>
                <port id="1">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="con" id="5" type="Concat" precision="FP32">
            <concat_data axis="1"/>
            <input>
                <port id="1">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>2</dim>
                    <dim>5</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>2</dim>
                    <dim>7</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="2"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="3" to-layer="5" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="5" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(gatherTF_test_params p) {
        std::string model = model_t;
        std::string inIdx;
        std::string inDict;
        std::string out;

        for (auto& idx : p.in_dim) {
            inIdx += "<dim>";
            inIdx += std::to_string(idx) + "</dim>\n";
        }

        for (auto& dct : p.dct_dim) {
            inDict += "<dim>";
            inDict += std::to_string(dct) + "</dim>\n";
        }

        for (auto& dst : p.ref_dim) {
            out += "<dim>";
            out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_OUTC_", inIdx);
        REPLACE_WITH_STR(model, "_IDICT_", inDict);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_STR(model, "_OUT_", out);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gatherTF_test_params p = ::testing::WithParamInterface<gatherTF_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            // Input Indexes
            InferenceEngine::Blob::Ptr srcIdx;
            int32_t in_size = 4;
            InferenceEngine::SizeVector in_dim = {2, 2};
            srcIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, in_dim, InferenceEngine::TensorDesc::getLayoutByDims(in_dim) });
            srcIdx->allocate();
            memcpy(static_cast<int32_t*>(srcIdx->buffer()), &p.in[0], sizeof(int32_t)*in_size);
            auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(srcIdx.get());
            if (srcIdxPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<int32_t>.";

            //  Input Dictionary
            InferenceEngine::Blob::Ptr srcDict = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.dct_dim, InferenceEngine::TensorDesc::getLayoutByDims(p.dct_dim) });
            srcDict->allocate();
            memcpy(srcDict->buffer(), &p.dct[0], sizeof(float)*p.dct.size());
            auto * srcDictPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcDict.get());
            if (srcDictPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            //  Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            //  Infer
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputDictionary", srcDict));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputText", srcIdx));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("Input3", srcIdx));
            graph.Infer(srcs, outputBlobs);

            //  Check results
            if (memcmp((*output).data(), &p.ref[0], p.ref.size()) != 0)
                FAIL() << "Wrong result with compare TF reference!";
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtGatherHolesTests, TestsGather) {}

INSTANTIATE_TEST_CASE_P(
    TestsGather, MKLDNNCPUExtGatherHolesTests,
    ::testing::Values(
        gatherTF_test_params{ { 1, 5, 2, 2 }, in1,{ 1, 3, 2, 2 }, dict, 1,{ 2, 2, 2, 2 }, ref_in1_a0_d322 }));

