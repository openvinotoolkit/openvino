// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"
#include "single_layer_common.hpp"

#include "test_graph.hpp"

#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct reshape_test_params {
    InferenceEngine::SizeVector in;
    InferenceEngine::SizeVector out;
    std::vector<size_t> shape;

    int axis;
    int num_axes;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_reshape(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for (int i=0; i < src.size(); i++)
        dst_data[i] = src_data[i];
}

class MKLDNNGraphReshapeTests: public TestsCommon,
                                     public WithParamInterface<reshape_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Reshape_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="norm" id="1" type="Reshape" precision="FP32">
            <data dim="_SHAPE_" axis="_AX_" num_axes="_NAX_"/>

            <input>
                <port id="1">
__SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(reshape_test_params p) {
        std::string model = model_t;

		std::string src_dims;
		for (auto& dim : p.in) {
			src_dims += "                    <dim>";
			src_dims += std::to_string(dim) + "</dim>\n";
		}
		REPLACE_WITH_STR(model, "__SRC_DIMS__", src_dims);

		std::string dst_dims;
		for (auto& dim : p.out) {
			dst_dims += "\t\t<dim>";
			dst_dims += std::to_string(dim) + "</dim>\n";
		}
		REPLACE_WITH_STR(model, "__DST_DIMS__", dst_dims);

        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_NUM(model, "_NAX_", p.num_axes);

        std::string shape_str;
        for (auto& dim : p.shape) {
            if (!shape_str.empty())
                shape_str += ",";
            shape_str += std::to_string(dim);
        }
        REPLACE_WITH_STR(model, "_SHAPE_", shape_str);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            reshape_test_params p = ::testing::WithParamInterface<reshape_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Reshape) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.in, InferenceEngine::ANY});
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_reshape(*srcPtr, dst_ref);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphReshapeTests, TestsReshape) {}


INSTANTIATE_TEST_CASE_P(
        TestsReshape, MKLDNNGraphReshapeTests,
        ::testing::Values(
        reshape_test_params{ {1, 3, 228, 228}, {1, 24, 2, 3249}, {1, 24, 2, 3249}, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown, { [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                ASSERT_EQ(1, impl.getConfig().inConfs.size());
                ASSERT_EQ(1, impl.getConfig().outConfs.size());
                ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4 },{ 2, 2 },{ 2, 2 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::C, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4 },{ 1, 2, 2 },{ 1, 2, 2 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::C, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::CHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4 },{ 1, 4, 1, 1 },{ 1, 4, 1, 1 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::C, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4, 4 },{ 1, 4, 4 },{ 1, 4, 4 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::CHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4, 4 },{ 1, 4, 2, 2 },{ 1, 4, 2, 2 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4, 2, 2 },{ 1, 4, 2, 2 },{ 1, 4, 2, 2 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::CHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 2, 2 },{ 4 },{ 4 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::C, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 1, 2, 2 },{ 4 },{ 4 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::CHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::C, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 1, 1, 2, 2 },{ 4 },{ 4 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::C, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4, 2, 2 },{ 4, 4 },{ 4, 4 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::CHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 1, 4, 2, 2 },{ 4, 4 },{ 4, 4 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 1, 4, 2, 2 },{ 4, 2, 2 },{ 4, 2, 2 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown,{ [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::CHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 1, 4, 2, 2 }, { 4, 2, 2, 1, 1 }, { 4, 2, 2, 1, 1 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown, { [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 4, 2, 2, 1, 1 }, { 1, 4, 2, 2 }, { 1, 4, 2, 2 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown, { [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } },
        reshape_test_params{ { 1, 200 }, { 1, 200, 1, 1, 1 }, { 1, 200, 1, 1, 1 }, 0, -1, 1,
            MKLDNNPlugin::impl_desc_type::unknown, { [](MKLDNNPlugin::PrimitiveDescInfo impl) {
            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
            ASSERT_EQ(1, impl.getConfig().inConfs.size());
            ASSERT_EQ(1, impl.getConfig().outConfs.size());
            ASSERT_EQ(InferenceEngine::Layout::NC, impl.getConfig().inConfs.at(0).desc.getLayout());
            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().outConfs.at(0).desc.getLayout());
        } } }
));
