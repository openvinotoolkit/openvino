// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct input_test_params {
    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNGraphInputTests: public TestsCommon,
                                     public WithParamInterface<input_test_params> {
    std::string model_t = R"V0G0N(
<net name="InputsOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="3">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="power1" id="4" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="1"/>
            <input>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="power2" id="5" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="1"/>
            <input>
                <port id="6">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="7">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="power3" id="6" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="1"/>
            <input>
                <port id="8">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="9">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="4"/>
        <edge from-layer="2" from-port="2" to-layer="5" to-port="6"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="8"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(input_test_params p) {
        return model_t;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            input_test_params p = ::testing::WithParamInterface<input_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Input || nodes[i]->getType() == MKLDNNPlugin::Output) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    size_t count = (nodes[i]->getType() == MKLDNNPlugin::Input) ? 0 : 2;
                    if (nodes[i]->getName() == "in3") {
                        count = 1;
                    }
                    if (nodes[i]->getName() == "out_power3") {
                        count = 3;
                    }
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(count)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphInputTests, TestsInput) {}


INSTANTIATE_TEST_CASE_P(
        TestsInput, MKLDNNGraphInputTests,
        ::testing::Values(
                input_test_params{1, MKLDNNPlugin::impl_desc_type::unknown, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(0, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        },
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(0, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().outConfs.at(0).desc.getLayout());
                        },
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(1, impl.getConfig().inConfs.size());
                            ASSERT_EQ(0, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                        },
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(1, impl.getConfig().inConfs.size());
                            ASSERT_EQ(0, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().inConfs.at(0).desc.getLayout());
                        }
                } }
        ));

class MKLDNNGraphConstInputTests: public TestsCommon {
    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="48"/>
            </blobs>
        </layer>
        <layer name="in2" type="Const" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="48" size="24"/>
            </blobs>
        </layer>
        <layer name="con" id="3" type="Concat" precision="FP32">
            <concat_data axis="2"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            std::string model = model_t;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, 
                {72}, InferenceEngine::C });
            weights->allocate();
            float * data = weights->buffer();

            std::cout << weights->size() << std::endl;

            InferenceEngine::SizeVector dims_src1 = {1, 3, 2, 2};
            InferenceEngine::SizeVector dims_src2 = {1, 3, 1, 2};
            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::NCHW});
            src1->allocate();
            float *srcData = src1->buffer();
            for (size_t i = 0; i < 12; i++, data++, srcData++) {
                *data = 1;
                *srcData = 1;
            }

            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::NCHW});
            src2->allocate();
            srcData = src2->buffer();
            for (size_t i = 0; i < 6; i++, data++, srcData++) {
                *data = 2;
                *srcData = 2;
            }
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            ASSERT_LE(3, nodes.size());

            InferenceEngine::BlobMap srcs;
            srcs["in1"] = src1;
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            // Compare
            float *src1_ptr = src1->buffer();
            size_t src1_size = src1->size();
            float *src2_ptr = src2->buffer();
            size_t src2_size = src2->size();
            float *dst_ptr = output->buffer();
            size_t dst_size = output->size();

            int len1 = 1, len2 = 1, cycles;
            for (int dim = 2; dim < output->getTensorDesc().getDims().size(); dim++) {
                len1 *= src1->getTensorDesc().getDims()[dim];
                len2 *= src2->getTensorDesc().getDims()[dim];
            }
            cycles = 2;

            int index1 = 0, index2 = 0, index = 0;
            for (int cycle = 0; cycle < cycles; cycle ++) {
                for (int i1 = 0; i1 < len1; i1++) {
                    if (src1_ptr[index1] != dst_ptr[index])
                    {
                        FAIL() << "index: " << index << " src: " << src1_ptr[index1] << ", dst: " << dst_ptr[index];
                    }
                    index1++; index++;
                }
                for (int i2 = 0; i2 < len2; i2++) {
                    if (src2_ptr[index2] != dst_ptr[index])
                    {
                        FAIL() << "index: " << index << " src: " << src2_ptr[index2] << ", dst: " << dst_ptr[index];
                    }
                    index2++; index++;
                }
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_F(MKLDNNGraphConstInputTests, TestsConstInput) {}


struct input_layout_test_params {
    InferenceEngine::Layout layout;
    std::vector<float> reference;
    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNGraphInputLayoutTest : public TestsCommon, public WithParamInterface<input_layout_test_params> {
    std::string model_t = R"V0G0N(
<net name="InputLayers" version="2" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="power1" id="1" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
    <pre-process reference-layer-name="input" mean-precision="FP32">
        <channel id="0">
            <mean value="1.0"/>
        </channel>
        <channel id="1">
            <mean value="2.0"/>
        </channel>
        <channel id="2">
            <mean value="3.0"/>
        </channel>
    </pre-process>
</net>
)V0G0N";

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            input_layout_test_params p = ::testing::WithParamInterface<input_layout_test_params>::GetParam();
            std::string model = model_t;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, { 1, 3, 2, 2 }, p.layout);
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
            src->allocate();
            fill_data_dbgval(src->buffer(), src->size());
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

            InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            InferenceEngine::BlobMap outputBlobs;
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);
            //  Check results
            if (memcmp((*output).data(), &p.reference[0], output->byteSize()) != 0)
                FAIL() << "Wrong result with compare reference!";
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphInputLayoutTest, TestsLayoutInput) {}

INSTANTIATE_TEST_CASE_P(
    TestsLayoutInput, MKLDNNGraphInputLayoutTest,
    ::testing::Values(
        input_layout_test_params{ InferenceEngine::NCHW, { 0,1,2,3,3,4,5,6,6,7,8,9 }, MKLDNNPlugin::impl_desc_type::unknown },
        input_layout_test_params{ InferenceEngine::NHWC, { 0,0,0,3,3,3,6,6,6,9,9,9 }, MKLDNNPlugin::impl_desc_type::unknown }
));

