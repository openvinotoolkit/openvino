// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <nodes/base.hpp>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace InferenceEngine;
using namespace Extensions;
using namespace ::Cpu;

struct permute_test_params {
    Layout layout_in, layout_out;
    Precision precision;
    size_t num_prim_desc;

    SizeVector dims;
    SizeVector permute_order;
    SizeVector block_dims_in;
    SizeVector block_order_in;
    SizeVector block_dims_out;
    SizeVector block_order_out;
};

class FakeLayerImpl_permute: public Cpu::ExtLayerBase,
                     public WithParamInterface<permute_test_params> {
public:
    explicit FakeLayerImpl_permute(const CNNLayer* layer) {
        try {
            layout = static_cast<Layout>(layer->GetParamAsUInt("layout"));
            block_dims = layer->GetParamAsInts("block_dims");
            order = layer->GetParamAsInts("order");
            addConfig(layer);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    Layout layout;
    std::vector<int> block_dims;
    std::vector<int> order;

    void addConfig(const CNNLayer* layer) {
        LayerConfig config;

        // Fill tensor parameters into config
        auto fill_port = [&] (std::vector<DataConfig>& port, const DataPtr& data) {
            if (!data) THROW_IE_EXCEPTION << "Cannot get input data!";

            DataConfig dataConfig;
            dataConfig.inPlace = 0;
            dataConfig.constant = false;

            const TensorDesc& data_desc = data->getTensorDesc();
            const SizeVector& data_dims = data_desc.getDims();

            InferenceEngine::Precision precision = data_desc.getPrecision();
            if (block_dims.empty()) {
                dataConfig.desc = TensorDesc(precision, data_dims, layout);
            } else {
                SizeVector tmp_block_dims(block_dims.size());
                SizeVector tmp_order(order.size());
                for (size_t i = 0; i < order.size(); i++) {
                    tmp_block_dims[i] = block_dims[i];
                    tmp_order[i] = order[i];
                }
                dataConfig.desc = TensorDesc(precision, data_dims, {tmp_block_dims, tmp_order});
            }

            port.push_back(dataConfig);
        };

        fill_port(config.inConfs, layer->insData[0].lock());
        fill_port(config.outConfs, layer->outData[0]);
        config.outConfs[0].desc.setPrecision(config.inConfs[0].desc.getPrecision());
        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        return OK;
    }
};

static std::string precToStr (Precision prec) {
    return prec == Precision::I8 ? "I8" : "FP32";
}

template <typename data_t>
static void fill_int_data(data_t *data, size_t size) {
    for (size_t i = 0 ; i < size; i++) {
        data[i] = i * 13 % 21 - 10;
    }
}

template <typename data_t>
static void ref_permute(const TBlob<data_t> &src, TBlob<float> &dst, permute_test_params prm) {
    const data_t *src_data = src.readOnly();
    float *dst_data = dst.data();

    SizeVector orderedDims;
    for (auto ord : prm.permute_order) {
        orderedDims.push_back(src.getTensorDesc().getDims()[ord]);
    }
    TensorDesc desc(Precision::FP32, src.getTensorDesc().getDims(), {orderedDims, prm.permute_order});

    for (int i=0; i < src.size(); i++) {
        dst_data[desc.offset(i)] = src_data[src.getTensorDesc().offset(i)];
    }
}

typedef std::tuple<Layout, Layout, Precision, size_t, SizeVector, SizeVector, SizeVector,
        SizeVector, SizeVector, SizeVector> test_params_t;

template <typename src_data_t>
class MKLDNNGraphPermuteTests: public TestsCommon,
public WithParamInterface<test_params_t> {
    std::string model_t = (std::string) R"V0G0N(
<Net Name="Permute_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="_PREC_" id="0">
            <output>
                <port id="0">
                    __DIMS__
                </port>
            </output>
        </layer>
        <layer name="fake1" type="FakeLayer_permute" precision="_PREC_" id="1">
            <data layout="_LAYOUT_IN_"
                  block_dims="_BLOCK_DIMS_IN_"
                  order="_BLOCK_ORDER_IN_"/>
            <input>
                <port id="0">
                    __DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    __DIMS__
                </port>
            </output>
        </layer>
        <layer name="permute" type="Permute" precision="_PREC_" id="2">
            <data order="_PERMUTE_ORDER_"/>
            <input>
                <port id="0">
                    __DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    __DST_DIMS__
                </port>
            </output>
        </layer>
        <layer name="fake2" type="FakeLayer_permute" precision="_PREC_" id="3">
            <data layout="_LAYOUT_OUT_"
                  block_dims="_BLOCK_DIMS_OUT_"
                  order="_BLOCK_ORDER_OUT_"/>
            <input>
                <port id="0">
                    __DST_DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    __DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
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
        for (auto& ord : p.permute_order) {
            if (!order.empty())
                order += ",";
            order += std::to_string(ord);
            dst_dims += "<dim>";
            dst_dims += std::to_string(p.dims[ord]) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "__DIMS__", dims);
        REPLACE_WITH_STR(model, "__DST_DIMS__", dst_dims);
        REPLACE_WITH_STR(model, "_PERMUTE_ORDER_", order);
        REPLACE_WITH_STR(model, "_PREC_", precToStr(p.precision));
        REPLACE_WITH_NUM(model, "_LAYOUT_IN_", (unsigned int)p.layout_in);
        REPLACE_WITH_NUM(model, "_LAYOUT_OUT_", (unsigned int)p.layout_out);

        REPLACE_WITH_NUM_VECTOR(model, "_BLOCK_DIMS_IN_", p.block_dims_in);
        REPLACE_WITH_NUM_VECTOR(model, "_BLOCK_ORDER_IN_", p.block_order_in);
        REPLACE_WITH_NUM_VECTOR(model, "_BLOCK_DIMS_OUT_", p.block_dims_out);
        REPLACE_WITH_NUM_VECTOR(model, "_BLOCK_ORDER_OUT_", p.block_order_out);

        return model;
    }

    virtual permute_test_params initialize_permute_test_params() {
        auto test_params = GetParam();
        permute_test_params p;

        p.layout_in = std::get<0>(test_params);
        p.layout_out = std::get<1>(test_params);
        p.precision = std::get<2>(test_params);
        p.num_prim_desc = std::get<3>(test_params);
        p.dims = std::get<4>(test_params);
        p.permute_order = std::get<5>(test_params);
        p.block_dims_in = std::get<6>(test_params);
        p.block_order_in = std::get<7>(test_params);
        p.block_dims_out = std::get<8>(test_params);
        p.block_order_out = std::get<9>(test_params);

        return p;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            permute_test_params p = initialize_permute_test_params();
            std::string model = getModel(p);

            Core core;
            CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            auto manager = std::make_shared<MKLDNNPlugin::MKLDNNExtensionManager>();
            {
                auto defaultExt = std::make_shared<Cpu::MKLDNNExtensions>();
                defaultExt->AddExt("FakeLayer_permute",
                    [](const CNNLayer* layer) -> InferenceEngine::ILayerImplFactory* {
                                    return new Cpu::ImplFactory<FakeLayerImpl_permute>(layer);
                                });
                manager->AddExtension(defaultExt);
            }
            graph.CreateGraph(network, manager);
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Permute) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                }
            }

            Blob::Ptr src = make_shared_blob<src_data_t>({p.precision, p.dims, TensorDesc::getLayoutByDims(p.dims)});
            src->allocate();
            if (typeid(src_data_t) == typeid(int8_t)) {
                fill_int_data(src->buffer().as<src_data_t *>(), src->size());
            } else {
                fill_data(src->buffer(), src->size());
            }

            auto* srcPtr = dynamic_cast<TBlob<src_data_t>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            BlobMap outputBlobs;

            auto item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            TensorDesc td(Precision::FP32, p.dims, TensorDesc::getLayoutByDims(p.dims));
            TBlob<float> dst_ref(td);
            dst_ref.allocate();

            ref_permute(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

using permute_f32 = MKLDNNGraphPermuteTests<float>;
using permute_s8 = MKLDNNGraphPermuteTests<int8_t>;

TEST_P(permute_f32, TestsPermute) {}
TEST_P(permute_s8, TestsPermute) {}

#define test_cases_planar_4d(prec) ::testing::Combine( \
        ::testing::Values(Layout::NCHW, Layout::NHWC), \
        ::testing::Values(Layout::NCHW, Layout::NHWC), \
        ::testing::Values(prec), \
        ::testing::Values(1 + (prec == Precision::I8)), \
        ::testing::Values(SizeVector({2, 3, 4, 5})), \
        ::testing::Values(SizeVector({0, 1, 2, 3}), SizeVector({0, 2, 3, 1}), \
                          SizeVector({0, 2, 1, 3}), SizeVector({0, 1, 3, 2}), \
                          SizeVector({1, 0, 2, 3})), \
        ::testing::Values(SizeVector({})), \
        ::testing::Values(SizeVector({})), \
        ::testing::Values(SizeVector({})), \
        ::testing::Values(SizeVector({})) \
)

#define test_cases_planar_5d(prec) ::testing::Combine( \
        ::testing::Values(Layout::NCDHW, Layout::NDHWC), \
        ::testing::Values(Layout::NCDHW, Layout::NDHWC), \
        ::testing::Values(prec), \
        ::testing::Values(1 + (prec == Precision::I8)), \
        ::testing::Values(SizeVector({2, 3, 4, 5, 6})), \
        ::testing::Values(SizeVector({0, 1, 2, 3, 4}), SizeVector({0, 4, 2, 1, 3}), \
                          SizeVector({0, 2, 4, 3, 1}), SizeVector({0, 3, 2, 4, 1}), \
                          SizeVector({0, 3, 1, 4, 2}), SizeVector({1, 0, 2, 3, 4})), \
        ::testing::Values(SizeVector({})), \
        ::testing::Values(SizeVector({})), \
        ::testing::Values(SizeVector({})), \
        ::testing::Values(SizeVector({})) \
)

#define case_planar_0(prec) test_params_t(Layout::NC, Layout::NC, prec, 1, {20, 3}, {0, 1}, {}, {}, {}, {})
#define case_planar_1(prec) test_params_t(Layout::CHW, Layout::CHW, prec, 1, {20, 30, 4}, {0, 1, 2}, {}, {}, {}, {})
#define case_planar_2(prec) test_params_t(Layout::CHW, Layout::CHW, prec, 1, {20, 30, 4}, {0, 2, 1}, {}, {}, {}, {})
#define case_planar_3(prec) test_params_t(Layout::CHW, Layout::CHW, prec, 1, {2, 12, 9}, {0, 2, 1}, {}, {}, {}, {})
#define case_planar_4(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 1, {2, 80, 2, 2, 4, 5}, {0, 1, 4, 2, 5, 3}, {}, {}, {}, {})
#define case_planar_5(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 1, {2, 8, 30, 3, 4, 5}, {0, 1, 4, 2, 5, 3}, {}, {}, {}, {})
#define case_planar_6(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 1, {2, 8, 3, 30, 4, 5}, {0, 3, 4, 1, 5, 2}, {}, {}, {}, {})

#define case_blocked_0(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 2, 3}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {2, 4, 10, 20, 8}, {0, 1, 2, 3, 1})
#define case_blocked_1(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 3, 1}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {2, 2, 20, 32, 8}, {0, 1, 2, 3, 1})
#define case_blocked_2(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 1, 3}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {2, 2, 32, 20, 8}, {0, 1, 2, 3, 1})
#define case_blocked_3(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 3, 2}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {2, 4, 20, 10, 8}, {0, 1, 2, 3, 1})
#define case_blocked_4(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 2 + (prec == Precision::I8), {10, 24, 4, 5}, {1, 0, 2, 3}, \
{10, 3, 4, 5, 8}, {0, 1, 2, 3, 1}, {24, 2, 4, 5, 8}, {0, 1, 2, 3, 1})
#define case_blocked_5(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 5, 10, 20}, {0, 1, 2, 3, 4}, \
{2, 4, 5, 10, 20, 8}, {0, 1, 2, 3, 4, 1}, {2, 4, 5, 10, 20, 8}, {0, 1, 2, 3, 4, 1})
#define case_blocked_6(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 5, 10, 20}, {0, 4, 2, 1, 3}, \
{2, 4, 5, 10, 20, 8}, {0, 1, 2, 3, 4, 1}, {2, 3, 5, 32, 10, 8}, {0, 1, 2, 3, 4, 1})
#define case_blocked_7(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 5, 10, 20}, {0, 2, 4, 3, 1}, \
{2, 4, 5, 10, 20, 8}, {0, 1, 2, 3, 4, 1}, {2, 1, 20, 10, 32, 8}, {0, 1, 2, 3, 4, 1})
#define case_blocked_8(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 5, 10, 20}, {0, 3, 2, 4, 1}, \
{2, 4, 5, 10, 20, 8}, {0, 1, 2, 3, 4, 1}, {2, 2, 5, 20, 32, 8}, {0, 1, 2, 3, 4, 1})
#define case_blocked_9(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 5, 10, 20}, {0, 3, 1, 4, 2}, \
{2, 4, 5, 10, 20, 8}, {0, 1, 2, 3, 4, 1}, {2, 2, 32, 20, 5, 8}, {0, 1, 2, 3, 4, 1})
#define case_blocked_10(prec) test_params_t(Layout::BLOCKED, Layout::BLOCKED, prec, 2 + (prec == Precision::I8), {10, 24, 4, 5, 6}, {1, 0, 2, 3, 4}, \
{10, 3, 4, 5, 6, 8}, {0, 1, 2, 3, 4, 1}, {24, 2, 4, 5, 6, 8}, {0, 1, 2, 3, 4, 1})

#define case_planar_to_blocked_0(prec) test_params_t(Layout::NCHW, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 2, 3}, \
{}, {}, {2, 4, 10, 20, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_1(prec) test_params_t(Layout::NCHW, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 3, 1}, \
{}, {}, {2, 2, 20, 32, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_2(prec) test_params_t(Layout::NCHW, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 1, 3}, \
{}, {}, {2, 2, 32, 20, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_3(prec) test_params_t(Layout::NCHW, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 3, 2}, \
{}, {}, {2, 4, 20, 10, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_4(prec) test_params_t(Layout::NCHW, Layout::BLOCKED, prec, 2 + (prec == Precision::I8), {10, 24, 4, 5}, {1, 0, 2, 3}, \
{}, {}, {24, 2, 4, 5, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_5(prec) test_params_t(Layout::NHWC, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 2, 3}, \
{}, {}, {2, 4, 10, 20, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_6(prec) test_params_t(Layout::NHWC, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 3, 1}, \
{}, {}, {2, 2, 20, 32, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_7(prec) test_params_t(Layout::NHWC, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 1, 3}, \
{}, {}, {2, 2, 32, 20, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_8(prec) test_params_t(Layout::NHWC, Layout::BLOCKED, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 3, 2}, \
{}, {}, {2, 4, 20, 10, 8}, {0, 1, 2, 3, 1})
#define case_planar_to_blocked_9(prec) test_params_t(Layout::NHWC, Layout::BLOCKED, prec, 2 + (prec == Precision::I8), {10, 24, 4, 5}, {1, 0, 2, 3}, \
{}, {}, {24, 2, 4, 5, 8}, {0, 1, 2, 3, 1})

#define case_blocked_to_planar_0(prec) test_params_t(Layout::BLOCKED, Layout::NCHW, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 2, 3}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_1(prec) test_params_t(Layout::BLOCKED, Layout::NCHW, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 3, 1}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_2(prec) test_params_t(Layout::BLOCKED, Layout::NCHW, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 1, 3}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_3(prec) test_params_t(Layout::BLOCKED, Layout::NCHW, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 3, 2}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_4(prec) test_params_t(Layout::BLOCKED, Layout::NCHW, prec, 2 + (prec == Precision::I8), {10, 24, 4, 5}, {1, 0, 2, 3}, \
{10, 3, 4, 5, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_5(prec) test_params_t(Layout::BLOCKED, Layout::NHWC, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 2, 3}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_6(prec) test_params_t(Layout::BLOCKED, Layout::NHWC, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 3, 1}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_7(prec) test_params_t(Layout::BLOCKED, Layout::NHWC, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 2, 1, 3}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_8(prec) test_params_t(Layout::BLOCKED, Layout::NHWC, prec, 3 + (prec == Precision::I8), {2, 32, 10, 20}, {0, 1, 3, 2}, \
{2, 4, 10, 20, 8}, {0, 1, 2, 3, 1}, {}, {})
#define case_blocked_to_planar_9(prec) test_params_t(Layout::BLOCKED, Layout::NHWC, prec, 2 + (prec == Precision::I8), {10, 24, 4, 5}, {1, 0, 2, 3}, \
{10, 3, 4, 5, 8}, {0, 1, 2, 3, 1}, {}, {})

test_params_t test_cases_fp32[] = {
        case_planar_0(Precision::FP32),
        case_planar_1(Precision::FP32),
        case_planar_2(Precision::FP32),
        case_planar_3(Precision::FP32),
        case_planar_4(Precision::FP32),
        case_planar_5(Precision::FP32),
        case_planar_6(Precision::FP32),
};

test_params_t test_cases_s8[] = {
        case_planar_0(Precision::I8),
        case_planar_1(Precision::I8),
        case_planar_2(Precision::I8),
        case_planar_3(Precision::I8),
        case_planar_4(Precision::I8),
        case_planar_5(Precision::I8),
        case_planar_6(Precision::I8),
};

test_params_t test_cases_blocked_fp32[] = {
        case_blocked_0(Precision::FP32),
        case_blocked_1(Precision::FP32),
        case_blocked_2(Precision::FP32),
        case_blocked_3(Precision::FP32),
        case_blocked_4(Precision::FP32),
        case_blocked_5(Precision::FP32),
        case_blocked_6(Precision::FP32),
        case_blocked_7(Precision::FP32),
        case_blocked_8(Precision::FP32),
        case_blocked_9(Precision::FP32),
        case_blocked_10(Precision::FP32),
};

test_params_t test_cases_blocked_s8[] = {
        case_blocked_0(Precision::I8),
        case_blocked_1(Precision::I8),
        case_blocked_2(Precision::I8),
        case_blocked_3(Precision::I8),
        case_blocked_4(Precision::I8),
        case_blocked_5(Precision::I8),
        case_blocked_6(Precision::I8),
        case_blocked_7(Precision::I8),
        case_blocked_8(Precision::I8),
        case_blocked_9(Precision::I8),
        case_blocked_10(Precision::I8),
};

test_params_t test_cases_planar_to_blocked_fp32[] = {
        case_planar_to_blocked_0(Precision::FP32),
        case_planar_to_blocked_1(Precision::FP32),
        case_planar_to_blocked_2(Precision::FP32),
        case_planar_to_blocked_3(Precision::FP32),
        case_planar_to_blocked_4(Precision::FP32),
        case_planar_to_blocked_5(Precision::FP32),
        case_planar_to_blocked_6(Precision::FP32),
        case_planar_to_blocked_7(Precision::FP32),
        case_planar_to_blocked_8(Precision::FP32),
        case_planar_to_blocked_9(Precision::FP32),
};

test_params_t test_cases_blocked_to_planar_fp32[] = {
        case_blocked_to_planar_0(Precision::FP32),
        case_blocked_to_planar_1(Precision::FP32),
        case_blocked_to_planar_2(Precision::FP32),
        case_blocked_to_planar_3(Precision::FP32),
        case_blocked_to_planar_4(Precision::FP32),
        case_blocked_to_planar_5(Precision::FP32),
        case_blocked_to_planar_6(Precision::FP32),
        case_blocked_to_planar_7(Precision::FP32),
        case_blocked_to_planar_8(Precision::FP32),
        case_blocked_to_planar_9(Precision::FP32),
};

test_params_t test_cases_planar_to_blocked_s8[] = {
        case_planar_to_blocked_0(Precision::I8),
        case_planar_to_blocked_1(Precision::I8),
        case_planar_to_blocked_2(Precision::I8),
        case_planar_to_blocked_3(Precision::I8),
        case_planar_to_blocked_4(Precision::I8),
        case_planar_to_blocked_5(Precision::I8),
        case_planar_to_blocked_6(Precision::I8),
        case_planar_to_blocked_7(Precision::I8),
        case_planar_to_blocked_8(Precision::I8),
        case_planar_to_blocked_9(Precision::I8),
};

test_params_t test_cases_blocked_to_planar_s8[] = {
        case_blocked_to_planar_0(Precision::I8),
        case_blocked_to_planar_1(Precision::I8),
        case_blocked_to_planar_2(Precision::I8),
        case_blocked_to_planar_3(Precision::I8),
        case_blocked_to_planar_4(Precision::I8),
        case_blocked_to_planar_5(Precision::I8),
        case_blocked_to_planar_6(Precision::I8),
        case_blocked_to_planar_7(Precision::I8),
        case_blocked_to_planar_8(Precision::I8),
        case_blocked_to_planar_9(Precision::I8),
};


INSTANTIATE_TEST_CASE_P(TestsPermutePlanar4d, permute_f32, test_cases_planar_4d(Precision::FP32));
INSTANTIATE_TEST_CASE_P(TestsPermutePlanar5d, permute_f32, test_cases_planar_5d(Precision::FP32));
INSTANTIATE_TEST_CASE_P(TestsPermute, permute_f32, ::testing::ValuesIn(test_cases_fp32));
INSTANTIATE_TEST_CASE_P(TestsPermuteBlocked, permute_f32, ::testing::ValuesIn(test_cases_blocked_fp32));
INSTANTIATE_TEST_CASE_P(TestsPermutePlanarToBlocked, permute_f32, ::testing::ValuesIn(test_cases_planar_to_blocked_fp32));
INSTANTIATE_TEST_CASE_P(TestsPermuteBlockedToPlanar, permute_f32, ::testing::ValuesIn(test_cases_blocked_to_planar_fp32));

INSTANTIATE_TEST_CASE_P(TestsPermutePlanar4d, permute_s8, test_cases_planar_4d(Precision::I8));
INSTANTIATE_TEST_CASE_P(TestsPermutePlanar5d, permute_s8, test_cases_planar_5d(Precision::I8));
INSTANTIATE_TEST_CASE_P(TestsPermute, permute_s8, ::testing::ValuesIn(test_cases_s8));
INSTANTIATE_TEST_CASE_P(TestsPermuteBlocked, permute_s8, ::testing::ValuesIn(test_cases_blocked_s8));
INSTANTIATE_TEST_CASE_P(TestsPermutePlanarToBlocked, permute_s8, ::testing::ValuesIn(test_cases_planar_to_blocked_s8));
INSTANTIATE_TEST_CASE_P(TestsPermuteBlockedToPlanar, permute_s8, ::testing::ValuesIn(test_cases_blocked_to_planar_s8));

class MKLDNNGraphDynBatchPermuteTests: public permute_f32 {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            permute_test_params p = initialize_permute_test_params();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
            if (MB < 2)
                MB = 2;
            p.dims[0] = MB;

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            auto manager = std::make_shared<MKLDNNPlugin::MKLDNNExtensionManager>();
            {
                auto defaultExt = std::make_shared<Cpu::MKLDNNExtensions>();
                defaultExt->AddExt("FakeLayer_permute",
                    [](const CNNLayer* layer) -> InferenceEngine::ILayerImplFactory* {
                                    return new Cpu::ImplFactory<FakeLayerImpl_permute>(layer);
                                });
                manager->AddExtension(defaultExt);
            }
            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(network, manager);

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.dims, InferenceEngine::TensorDesc::getLayoutByDims(p.dims)});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
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

test_params_t test_cases_dyn_batch[] = {
        test_params_t(Layout::NCHW, Layout::NCHW, Precision::FP32, 1, {2, 3, 4, 5}, {0, 1, 2, 3}, {}, {}, {}, {}),
        test_params_t(Layout::NCHW, Layout::NCHW, Precision::FP32, 1, {2, 3, 4, 5}, {0, 2, 3, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NCHW, Layout::NCHW, Precision::FP32, 1, {2, 3, 4, 5}, {0, 2, 1, 3}, {}, {}, {}, {}),
        test_params_t(Layout::CHW, Layout::CHW, Precision::FP32, 1, {2, 3, 4}, {0, 1, 2}, {}, {}, {}, {}),
        test_params_t(Layout::CHW, Layout::CHW, Precision::FP32, 1, {2, 3, 4}, {0, 2, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NC, Layout::NC, Precision::FP32, 1, {2, 3}, {0, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 6}, {0, 4, 2, 1, 3}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 6}, {0, 2, 4, 3, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 6}, {0, 3, 2, 4, 1}, {}, {}, {}, {}),
        // FIXME: Plugin inserts reorder from blocked to goidhw format here
        // test_params_t(Layout::BLOCKED, Layout::BLOCKED, Precision::FP32, 1, {2, 8, 2, 2, 4, 5}, {0, 1, 4, 2, 5, 3}, {}, {}, {}, {}),
        // test_params_t(Layout::BLOCKED, Layout::BLOCKED, Precision::FP32, 1, {2, 8, 3, 3, 4, 5}, {0, 1, 4, 2, 5, 3}, {}, {}, {}, {}),
        test_params_t(Layout::CHW, Layout::CHW, Precision::FP32, 1, {2, 12, 9}, {0, 2, 1}, {}, {}, {}, {}),
        // test_params_t(Layout::BLOCKED, Layout::BLOCKED, Precision::FP32, 1, {2, 8, 3, 3, 4, 5}, {0, 3, 4, 1, 5, 2}, {}, {}, {}, {}),
        test_params_t(Layout::NCHW, Layout::NCHW, Precision::FP32, 1, {2, 3, 4, 5}, {0, 1, 3, 2}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 7}, {0, 3, 1, 4, 2}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 7}, {0, 2, 1, 3, 4}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 7}, {0, 2, 4, 3, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {2, 3, 4, 5, 7}, {0, 4, 2, 3, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NCHW, Layout::NCHW, Precision::FP32, 1, {2, 3, 4, 5}, {0, 3, 1, 2}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {3, 4, 7, 8, 4}, {0, 2, 3, 4, 1}, {}, {}, {}, {}),
        test_params_t(Layout::NCDHW, Layout::NCDHW, Precision::FP32, 1, {3, 4, 7, 8, 4}, {0, 4, 1, 2, 3}, {}, {}, {}, {}),
};

INSTANTIATE_TEST_CASE_P(TestsDynBatchPermute, MKLDNNGraphDynBatchPermuteTests, ::testing::ValuesIn(test_cases_dyn_batch));
