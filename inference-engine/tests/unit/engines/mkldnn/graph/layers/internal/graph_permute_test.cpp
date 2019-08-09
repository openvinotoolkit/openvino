// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <inference_engine/cnn_network_impl.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct permute_test_params {
    InferenceEngine::SizeVector dims;
    InferenceEngine::SizeVector order;
    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_permute(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, permute_test_params prm) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    InferenceEngine::SizeVector orderedDims;
    for (auto ord : prm.order) {
        orderedDims.push_back(src.getTensorDesc().getDims()[ord]);
    }
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, src.getTensorDesc().getDims(), {orderedDims, prm.order});

    for (int i=0; i < src.size(); i++) {
        dst_data[desc.offset(i)] = src_data[src.getTensorDesc().offset(i)];
    }
}

class MKLDNNGraphPermuteTests: public TestsCommon,
                               public WithParamInterface<permute_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Power_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    __DIMS__
                </port>
            </output>
        </layer>
        <layer name="permute" id="1" type="Permute" precision="FP32">
            <data order="_ORDER_"/>
            <input>
                <port id="1">
                    __DIMS__
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

protected:
    std::string getModel(permute_test_params p) {
        std::string model = model_t;
        std::string dims;
        std::string dst_dims;
        for (auto& dim : p.dims) {
            dims += "<dim>";
            dims += std::to_string(dim) + "</dim>\n";
        }

        std::string order;
        for (auto& ord : p.order) {
            if (!order.empty())
                order += ",";
            order += std::to_string(ord);
            dst_dims += "<dim>";
            dst_dims += std::to_string(p.dims[ord]) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "__DIMS__", dims);
        REPLACE_WITH_STR(model, "__DST_DIMS__", dst_dims);
        REPLACE_WITH_STR(model, "_ORDER_", order);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            permute_test_params p = ::testing::WithParamInterface<permute_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Permute) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.dims, InferenceEngine::TensorDesc::getLayoutByDims(p.dims)});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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

            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, p.dims, InferenceEngine::TensorDesc::getLayoutByDims(p.dims));
            InferenceEngine::TBlob<float> dst_ref(td);
            dst_ref.allocate();

            ref_permute(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphPermuteTests, TestsPermute) {}

INSTANTIATE_TEST_CASE_P(
        TestsPermute, MKLDNNGraphPermuteTests,
        ::testing::Values(
                permute_test_params{{2, 3, 4, 5}, {0, 1, 2, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 2, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {3, 0, 1, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {1, 3, 2, 0}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {3, 2, 1, 0}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 2, 1, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {0, 1, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {0, 2, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {2, 1, 0}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {1, 2, 0}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {2, 0, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {1, 0, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3}, {1, 0}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3}, {0, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 4, 2, 1, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 2, 4, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 3, 2, 4, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 2, 2, 4, 5}, {0, 1, 4, 2, 5, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 3, 3, 4, 5}, {0, 1, 4, 2, 5, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 3, 4}, {3, 0, 1, 2}, 2, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 12, 9}, {0, 2, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 3, 3, 4, 5}, {0, 3, 4, 1, 5, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 1, 3, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 3, 1, 4, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {1, 2, 0, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 2, 1, 3, 4}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 2, 4, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 4, 2, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 3, 1, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{3, 4, 7}, {1, 0, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown}
        ));

class MKLDNNGraphDynBatchPermuteTests: public MKLDNNGraphPermuteTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            permute_test_params p = ::testing::WithParamInterface<permute_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
            if (MB < 2)
                MB = 2;
            p.dims[0] = MB;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.dims, InferenceEngine::TensorDesc::getLayoutByDims(p.dims)});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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

            auto checkPermute = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Permute;
            };
            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkPermute);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkPermute);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchPermuteTests, TestsDynBatchPermute) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchPermute, MKLDNNGraphDynBatchPermuteTests,
        ::testing::Values(
                permute_test_params{{2, 3, 4, 5}, {0, 1, 2, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 2, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 2, 1, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {0, 1, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4}, {0, 2, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3}, {0, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 4, 2, 1, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 2, 4, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 6}, {0, 3, 2, 4, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 2, 2, 4, 5}, {0, 1, 4, 2, 5, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 3, 3, 4, 5}, {0, 1, 4, 2, 5, 3}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 12, 9}, {0, 2, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 8, 3, 3, 4, 5}, {0, 3, 4, 1, 5, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 1, 3, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 3, 1, 4, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 2, 1, 3, 4}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 2, 4, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5, 7}, {0, 4, 2, 3, 1}, 1, MKLDNNPlugin::impl_desc_type::unknown},
                permute_test_params{{2, 3, 4, 5}, {0, 3, 1, 2}, 1, MKLDNNPlugin::impl_desc_type::unknown}
        ));
