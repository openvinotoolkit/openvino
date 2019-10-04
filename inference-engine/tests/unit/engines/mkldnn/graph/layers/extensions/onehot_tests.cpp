// Copyright (C) 2018-2019 Intel Corporation
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

#include "single_layer_common.hpp"
#include "cpp/ie_cnn_net_reader.h"

using namespace ::testing;
using namespace InferenceEngine;

struct one_hot_base_params {
    struct { size_t n, c, h, w; } in;
    struct { size_t d, n, c, h, w; } out;
    int axis;
    unsigned int depth;
    float on, off;
};

struct one_hot_test_params : one_hot_base_params {
    std::string libraryName;
    TargetDevice targetDevice;

    one_hot_test_params(std::string name, one_hot_base_params params,
                        TargetDevice targetDevice = TargetDevice::eDefault) :
            one_hot_base_params(params), libraryName(name), targetDevice(targetDevice) {}
};

class OneHotOnly1dTest: public TestsCommon,
                       public WithParamInterface<one_hot_test_params> {

    std::string model_t = R"V0G0N(
<net name="OneHot_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="OneHot1" type="OneHot" precision="FP32">

            <data depth="_DEPTH_" axis="_AXIS_"/>

            <input>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>l
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(one_hot_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_DEPTH_", p.depth);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        return model;
    }
    void ref_one_hot_1d(InferenceEngine::Blob &src, InferenceEngine::Blob &dst, one_hot_test_params p)
    {
        float *src_ptr = src.buffer().as<float*>();
        std::size_t src_size = src.size();
        float *dst_ptr = dst.buffer().as<float*>();
        std::size_t dst_size = dst.size();

        for (int ow = 0; ow < p.out.w; ow++) {
            std::size_t src_offset = 0;
            std::size_t dst_offset = ow;

            int hot_axis = -1;
            if (p.axis == -1) {
                hot_axis = ow;
                src_offset = 0;
            } else if (p.axis == 0) {
                hot_axis = ow;
                src_offset = 0;
            }
            int v = src_ptr[src_offset];

            dst_ptr[dst_offset] = (v == hot_axis) ? p.on : p.off;
        }
    }

protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            one_hot_test_params p = ::testing::WithParamInterface<one_hot_test_params>::GetParam();
            std::string model = getModel(p);

            CNNNetReader net_reader;
            try {
                net_reader.ReadNetwork(model.data(), model.length());
            } catch (InferenceEngine::details::InferenceEngineException &e) {
                FAIL() << e.what();
            } catch (std::exception &e) {
                FAIL() << e.what();
            }

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

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

            SizeVector dims_src = {};
            TBlob<float> src({Precision::FP32, dims_src, Layout::SCALAR});
            src.allocate();
            float * s = src.buffer().as<float*>();
            s[0] = 2;

            ref_one_hot_1d(src, dst_ref, p);

            InferenceEngine::Blob::Ptr pSrc = make_shared_blob<float>(src);
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", pSrc));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};



class OneHotOnly2dTest: public TestsCommon,
                       public WithParamInterface<one_hot_test_params> {

    std::string model_t = R"V0G0N(
<net name="OneHot_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="OneHot1" type="OneHot" precision="FP32">

            <data depth="_DEPTH_" axis="_AXIS_"/>

            <input>
                <port id="1">
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>l
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(one_hot_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_DEPTH_", p.depth);

        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        return model;
    }
    void ref_one_hot_2d(InferenceEngine::Blob &src, InferenceEngine::Blob &dst, one_hot_test_params p)
    {
        float *src_ptr = src.buffer().as<float*>();
        std::size_t src_size = src.size();
        float *dst_ptr = dst.buffer().as<float*>();
        std::size_t dst_size = dst.size();

        for (int oh = 0; oh < p.out.h; oh++) {
            for (int ow = 0; ow < p.out.w; ow++) {
                std::size_t src_offset = 0;

                std::size_t dst_offset = ow + p.out.w * oh;

                int hot_axis = -1;
                if (p.axis == -1) {
                    hot_axis = ow;
                    src_offset = oh;
                } else if (p.axis == 0) {
                    hot_axis = oh;
                    src_offset = ow;
                } else if (p.axis == 1) {
                    hot_axis = ow;
                    src_offset = oh;
                }
                int v = src_ptr[src_offset];

                dst_ptr[dst_offset] = (v == hot_axis) ? p.on : p.off;
            }
        }
    }

protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            one_hot_test_params p = ::testing::WithParamInterface<one_hot_test_params>::GetParam();
            std::string model = getModel(p);

            CNNNetReader net_reader;
            try {
                net_reader.ReadNetwork(model.data(), model.length());
            } catch (InferenceEngine::details::InferenceEngineException &e) {
                FAIL() << e.what();
            } catch (std::exception &e) {
                FAIL() << e.what();
            }

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

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

            SizeVector dims_src = {p.in.w};
            TBlob<float> src({Precision::FP32, dims_src, Layout::C});
            src.allocate();
            float * s = src.buffer().as<float*>();
            for (int i = 0; i < src.size(); ++i)
                s[i] = -1;
            s[0] = 3;
            s[2] = 2;

            // Check results
            InferenceEngine::SizeVector out_dims = {p.out.w, p.out.h};
            ref_one_hot_2d(src, dst_ref, p);

            InferenceEngine::Blob::Ptr pSrc = make_shared_blob<float>(src);
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", pSrc));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};


class OneHotOnly3dTest: public TestsCommon,
                       public WithParamInterface<one_hot_test_params> {

    std::string model_t = R"V0G0N(
<net name="OneHot_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="OneHot1" type="OneHot" precision="FP32">

            <data depth="_DEPTH_" axis="_AXIS_" on_value="_ON_VALUE_" off_value="_OFF_VALUE_"/>

            <input>
                <port id="1">
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>l
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(one_hot_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_DEPTH_", p.depth);
        REPLACE_WITH_NUM(model, "_ON_VALUE_", p.on);
        REPLACE_WITH_NUM(model, "_OFF_VALUE_", p.off);

        REPLACE_WITH_NUM(model, "_OC_", p.out.c);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        return model;
    }
    void ref_one_hot_3d(InferenceEngine::Blob &src, InferenceEngine::Blob &dst, one_hot_test_params p)
    {
        float *src_ptr = src.buffer().as<float*>();
        std::size_t src_size = src.size();
        float *dst_ptr = dst.buffer().as<float*>();
        std::size_t dst_size = dst.size();

        for (int oc = 0; oc < p.out.c; oc++) {
            for (int oh = 0; oh < p.out.h; oh++) {
                for (int ow = 0; ow < p.out.w; ow++) {
                    std::size_t src_offset = 0;

                    std::size_t dst_offset = ow + p.out.w * oh + p.out.w * p.out.h * oc;

                    int hot_axis = -1;
                    if (p.axis == -1) {
                        hot_axis = ow;
                        src_offset = oh + p.in.w * oc;
                    } else if (p.axis == 0) {
                        hot_axis = oc;
                        src_offset = ow + p.in.w * oh;
                    } else if (p.axis == 1) {
                        hot_axis = oh;
                        src_offset = ow + p.in.w * oc;
                    }
                    int v = src_ptr[src_offset];

                    dst_ptr[dst_offset] = (v == hot_axis) ? p.on : p.off;
                }
            }
        }
    }

protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            one_hot_test_params p = ::testing::WithParamInterface<one_hot_test_params>::GetParam();
            std::string model = getModel(p);

            CNNNetReader net_reader;
            try {
                net_reader.ReadNetwork(model.data(), model.length());
            } catch (InferenceEngine::details::InferenceEngineException &e) {
                FAIL() << e.what();
            } catch (std::exception &e) {
                FAIL() << e.what();
            }

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

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

            SizeVector dims_src = {p.in.w, p.in.h};
            TBlob<float> src({Precision::FP32, dims_src, Layout::HW});
            src.allocate();
            float * s = src.buffer().as<float*>();
            for (int i = 0; i < src.size(); ++i)
                s[i] = -1;
            s[0] = 3;
            s[4] = 2;

            // Check results
            InferenceEngine::SizeVector out_dims = {p.out.w, p.out.h, p.out.c};
            ref_one_hot_3d(src, dst_ref, p);

            InferenceEngine::Blob::Ptr pSrc = make_shared_blob<float>(src);
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", pSrc));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

class OneHotOnly4dTest: public TestsCommon,
                       public WithParamInterface<one_hot_test_params> {

    std::string model_t = R"V0G0N(
<net name="OneHot_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="OneHot1" type="OneHot" precision="FP32">

            <data depth="_DEPTH_" axis="_AXIS_"/>

            <input>
                <port id="1">
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_ON_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>l
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(one_hot_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_DEPTH_", p.depth);

        REPLACE_WITH_NUM(model, "_ON_", p.out.n);
        REPLACE_WITH_NUM(model, "_OC_", p.out.c);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        return model;
    }
void ref_one_hot_4d(InferenceEngine::Blob &src, InferenceEngine::Blob &dst, one_hot_test_params p)
{
    float *src_ptr = src.buffer().as<float*>();
    std::size_t src_size = src.size();
    float *dst_ptr = dst.buffer().as<float*>();
    std::size_t dst_size = dst.size();

    for (int ob = 0; ob < p.out.n; ob++) {
        for (int oc = 0; oc < p.out.c; oc++) {
            for (int oh = 0; oh < p.out.h; oh++) {
                for (int ow = 0; ow < p.out.w; ow++) {
                    std::size_t src_offset = 0;

                    std::size_t dst_offset = ow + p.out.w * oh + p.out.w * p.out.h * oc + p.out.w * p.out.h * p.out.c * ob;

                    int hot_axis = -1;
                    if (p.axis == -1) {
                        hot_axis = ow;
                        src_offset = oh + p.in.w * oc + p.in.w * p.in.h * ob;
                    } else if (p.axis == 0) {
                        hot_axis = ob;
                        src_offset = ow + p.in.w * oh + p.in.w * p.in.h * oc;
                    } else if (p.axis == 1) {
                        hot_axis = oc;
                        src_offset = ow + p.in.w * oh + p.in.w * p.in.h * ob;
                    } else if (p.axis == 2) {
                        hot_axis = oh;
                        src_offset = ow + p.in.w * oc + p.in.w * p.in.h * ob;
                    }
                    int v = src_ptr[src_offset];

                    dst_ptr[dst_offset] = (v == hot_axis) ? p.on : p.off;
                }
            }
        }
    }
}
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            one_hot_test_params p = ::testing::WithParamInterface<one_hot_test_params>::GetParam();
            std::string model = getModel(p);

            CNNNetReader net_reader;
            try {
                net_reader.ReadNetwork(model.data(), model.length());
            } catch (InferenceEngine::details::InferenceEngineException &e) {
                FAIL() << e.what();
            } catch (std::exception &e) {
                FAIL() << e.what();
            }

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

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

            SizeVector dims_src = {p.in.w, p.in.h, p.in.c};

            TBlob<float> src({Precision::FP32, dims_src, Layout::CHW});
            src.allocate();

            float * s = src.buffer().as<float*>();
            for (int i = 0; i < src.size(); ++i)
                s[i] = -1;
            s[0] = 3;
            s[4] = 2;

            // Check results
            InferenceEngine::SizeVector out_dims = {p.out.w, p.out.h, p.out.c, p.out.n};
            ref_one_hot_4d(src, dst_ref, p);

            InferenceEngine::Blob::Ptr pSrc = make_shared_blob<float>(src);
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", pSrc));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};


class OneHotOnly5dTest: public TestsCommon,
                       public WithParamInterface<one_hot_test_params> {

    std::string model_t = R"V0G0N(
<net name="OneHot_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="OneHot1" type="OneHot" precision="FP32">

            <data depth="_DEPTH_" axis="_AXIS_"/>

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_ON_</dim>
                    <dim>_OC_</dim>
                    <dim>_OD_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>l
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(one_hot_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_DEPTH_", p.depth);

        REPLACE_WITH_NUM(model, "_ON_", p.out.n);
        REPLACE_WITH_NUM(model, "_OC_", p.out.c);
        REPLACE_WITH_NUM(model, "_OD_", p.out.d);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        return model;
    }
void ref_one_hot_5d(InferenceEngine::Blob &src, InferenceEngine::Blob &dst, one_hot_test_params p)
{
    float *src_ptr = src.buffer().as<float*>();
    std::size_t src_size = src.size();
    float *dst_ptr = dst.buffer().as<float*>();
    std::size_t dst_size = dst.size();

    for (int ob = 0; ob < p.out.n; ob++) {
        for (int oc = 0; oc < p.out.c; oc++) {
            for (int od = 0; od < p.out.d; od++) {
                for (int oh = 0; oh < p.out.h; oh++) {
                    for (int ow = 0; ow < p.out.w; ow++) {
                        std::size_t src_offset = 0;

                        std::size_t dst_offset = ow + p.out.w * oh + p.out.w * p.out.h * od \
                            + p.out.w * p.out.h * p.out.d * oc  + p.out.w * p.out.h * p.out.d * p.out.c * ob;

                        int hot_axis = -1;
                        if (p.axis == -1 || p.axis == 4) {
                            hot_axis = ow;
                            src_offset = oh + p.in.w * od + p.in.w * p.in.h * oc + p.in.w * p.in.h * p.in.c * ob;
                        } else if (p.axis == 0) {
                            hot_axis = ob;
                            src_offset = ow + p.in.w * oh + p.in.w * p.in.h * od + p.in.w * p.in.h * p.in.c * oc;
                        } else if (p.axis == 1) {
                            hot_axis = oc;
                            src_offset = ow + p.in.w * oh + p.in.w * p.in.h * od + p.in.w * p.in.h * p.in.c * ob;
                        } else if (p.axis == 2) {
                            hot_axis = od;
                            src_offset = ow + p.in.w * oh + p.in.w * p.in.h * oc + p.in.w * p.in.h * p.in.c * ob;
                        } else if (p.axis == 3) {
                            hot_axis = oh;
                            src_offset = ow + p.in.w * od + p.in.w * p.in.h * oc + p.in.w * p.in.h * p.in.c * ob;
                        }

                        int v = src_ptr[src_offset];
                        dst_ptr[dst_offset] = (v == hot_axis) ? p.on : p.off;
                    }
                }
            }
        }
    }
}
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            one_hot_test_params p = ::testing::WithParamInterface<one_hot_test_params>::GetParam();
            std::string model = getModel(p);

            CNNNetReader net_reader;
            try {
                net_reader.ReadNetwork(model.data(), model.length());
            } catch (InferenceEngine::details::InferenceEngineException &e) {
                FAIL() << e.what();
            } catch (std::exception &e) {
                FAIL() << e.what();
            }

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

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

            SizeVector dims_src = {p.in.w, p.in.h, p.in.c, p.in.n};

            TBlob<float> src({Precision::FP32, dims_src, Layout::NCHW});
            src.allocate();

            float * s = src.buffer().as<float*>();
            for (int i = 0; i < src.size(); ++i)
                s[i] = -1;
            s[3] = 3;
            s[7] = 2;



            // Check results
            ref_one_hot_5d(src, dst_ref, p);

            InferenceEngine::Blob::Ptr pSrc = make_shared_blob<float>(src);
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", pSrc));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

// 0d -> 1d, depth
#define case_1d_0 one_hot_base_params({ {0, 0, 0, 0}, {0, 0, 0, 0, 3},-1, 3, 1.0f, 0.0f })
#define case_1d_1 one_hot_base_params({ {0, 0, 0, 0}, {0, 0, 0, 0, 4}, 0, 4, 1.0f, 0.0f })
// 1d -> 2d, axis default
#define case_2d_0 one_hot_base_params({ {0, 0, 0, 3}, {0, 0, 0, 3, 6},-1, 6, 1.0f, 0.0f })
#define case_2d_1 one_hot_base_params({ {0, 0, 0, 3}, {0, 0, 0, 6, 3}, 0, 6, 1.0f, 0.0f })
#define case_2d_2 one_hot_base_params({ {0, 0, 0, 3}, {0, 0, 0, 3, 6}, 1, 6, 1.0f, 0.0f })
// 2d -> 3d, on_value, off_value
#define case_3d_0 one_hot_base_params({ {0, 0, 3, 2}, {0, 0, 3, 2, 4},-1, 4, 2.0f, -1.0f })
#define case_3d_1 one_hot_base_params({ {0, 0, 3, 2}, {0, 0, 4, 3, 2}, 0, 4, 2.0f, -1.0f })
#define case_3d_2 one_hot_base_params({ {0, 0, 3, 2}, {0, 0, 3, 4, 2}, 1, 4, 2.0f, -1.0f })
// 3d -> 4d
#define case_4d_0 one_hot_base_params({ {0, 1, 3, 2}, {0, 1, 3, 2, 4},-1, 4, 1.0f, 0.0f })
#define case_4d_1 one_hot_base_params({ {0, 1, 3, 2}, {0, 4, 1, 3, 2}, 0, 4, 1.0f, 0.0f })
#define case_4d_2 one_hot_base_params({ {0, 1, 3, 2}, {0, 1, 4, 3, 2}, 1, 4, 1.0f, 0.0f })
#define case_4d_3 one_hot_base_params({ {0, 1, 3, 2}, {0, 1, 3, 4, 2}, 2, 4, 1.0f, 0.0f })
// 4d -> 5d IE layouts are NCHW -> NCDHW, param layouts are {n, c , h, w} {d, n, c, h ,w}
#define case_5d_0 one_hot_base_params({ {1, 3, 2, 3}, {2, 1, 3, 3, 4},-1, 4, 1.0f, 0.0f })
#define case_5d_1 one_hot_base_params({ {1, 3, 2, 3}, {3, 4, 1, 2, 3}, 0, 4, 1.0f, 0.0f })
#define case_5d_2 one_hot_base_params({ {1, 3, 2, 3}, {3, 1, 4, 2, 3}, 1, 4, 1.0f, 0.0f })
#define case_5d_3 one_hot_base_params({ {1, 3, 2, 3}, {4, 1, 3, 2, 3}, 2, 4, 1.0f, 0.0f })
#define case_5d_4 one_hot_base_params({ {1, 3, 2, 3}, {2, 1, 3, 4, 3}, 3, 4, 1.0f, 0.0f })

one_hot_test_params one_hot_only_1d_test_cases[] = {
    one_hot_test_params("MKLDNNPlugin", case_1d_0),
    one_hot_test_params("MKLDNNPlugin", case_1d_1)
};

one_hot_test_params one_hot_only_2d_test_cases[] = {
    one_hot_test_params("MKLDNNPlugin", case_2d_0),
    one_hot_test_params("MKLDNNPlugin", case_2d_1),
    one_hot_test_params("MKLDNNPlugin", case_2d_2),
};

one_hot_test_params one_hot_only_3d_test_cases[] = {
    one_hot_test_params("MKLDNNPlugin", case_3d_0),
    one_hot_test_params("MKLDNNPlugin", case_3d_1),
    one_hot_test_params("MKLDNNPlugin", case_3d_2),
};
one_hot_test_params one_hot_only_4d_test_cases[] = {
    one_hot_test_params("MKLDNNPlugin", case_4d_0),
    one_hot_test_params("MKLDNNPlugin", case_4d_1),
    one_hot_test_params("MKLDNNPlugin", case_4d_2),
    one_hot_test_params("MKLDNNPlugin", case_4d_3)
};

one_hot_test_params one_hot_only_5d_test_cases[] = {
    one_hot_test_params("MKLDNNPlugin", case_5d_0),
    one_hot_test_params("MKLDNNPlugin", case_5d_1),
    one_hot_test_params("MKLDNNPlugin", case_5d_2),
    one_hot_test_params("MKLDNNPlugin", case_5d_3),
    one_hot_test_params("MKLDNNPlugin", case_5d_4)
};

TEST_P(OneHotOnly1dTest, TestsOneHot) {}
INSTANTIATE_TEST_CASE_P(TestsOneHot, OneHotOnly1dTest, ::testing::ValuesIn(one_hot_only_1d_test_cases));

TEST_P(OneHotOnly2dTest, TestsOneHot) {}
INSTANTIATE_TEST_CASE_P(TestsOneHot, OneHotOnly2dTest, ::testing::ValuesIn(one_hot_only_2d_test_cases));

TEST_P(OneHotOnly3dTest, TestsOneHot) {}
INSTANTIATE_TEST_CASE_P(TestsOneHot, OneHotOnly3dTest, ::testing::ValuesIn(one_hot_only_3d_test_cases));

TEST_P(OneHotOnly4dTest, TestsOneHot) {}
INSTANTIATE_TEST_CASE_P(TestsOneHot, OneHotOnly4dTest, ::testing::ValuesIn(one_hot_only_4d_test_cases));

TEST_P(OneHotOnly5dTest, TestsOneHot) {}
INSTANTIATE_TEST_CASE_P(TestsOneHot, OneHotOnly5dTest, ::testing::ValuesIn(one_hot_only_5d_test_cases));

