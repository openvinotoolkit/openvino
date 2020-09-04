// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <tests_common_func.hpp>
#include <memory>
#include <multi-device/multi_device_config.hpp>
#include <ie_core.hpp>
#include <ie_plugin_ptr.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/ngraph.hpp>

using namespace ::testing;
using namespace InferenceEngine;

struct extension_params {
    std::string pluginName;
    std::shared_ptr<IExtension> extension;
    std::string plugin() { return pluginName + "Plugin"; }
    // optional config (used for multi-device)
    std::map<std::string, std::string> config;
};

class FakePrimitiveImpl : public InferenceEngine::ILayerExecImpl {
public:
    FakePrimitiveImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        if (cnnLayer->outData.size() != 1 && cnnLayer->insData.size() != 1)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = false;
        cfg.inPlace = 0;
        InferenceEngine::SizeVector order;
        for(size_t i = 0; i < cnnLayer->outData[0]->getTensorDesc().getDims().size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->outData[0]->getTensorDesc().getPrecision(),
                                               cnnLayer->outData[0]->getTensorDesc().getDims(),
                                               {cnnLayer->outData[0]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(cfg);
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer* cnnLayer;
};

class TestExtension : public InferenceEngine::IExtension {
public:
    void Release() noexcept override { delete this; }

    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override
    {
        static const InferenceEngine::Version VERSION{{}, "", ""};
        versionInfo = &VERSION;
    }

    void Unload() noexcept override {}
};

class NewFakePrimitiveImpl : public InferenceEngine::ILayerExecImpl {
public:
    NewFakePrimitiveImpl(const std::shared_ptr<ngraph::Node>& node): node(node) {}

    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        if (node->outputs().size() != 1 && node->inputs().size() != 1)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = false;
        cfg.inPlace = 0;
        InferenceEngine::SizeVector order;
        auto partialShape = node->get_output_partial_shape(0);
        if (partialShape.is_dynamic())
            return InferenceEngine::GENERAL_ERROR;
        auto shape = node->get_output_shape(0);
        for(size_t i = 0; i < shape.size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                               shape, {shape, order});
        config.outConfs.push_back(cfg);
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }

private:
    const std::shared_ptr<ngraph::Node> node;
};

class FakeTestOp: public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"Fake", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    FakeTestOp() = default;
    explicit FakeTestOp(const ngraph::Output<ngraph::Node>& arg): Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_shape = get_input_partial_shape(0).to_shape();

        ngraph::Shape output_shape(input_shape);
        for (int i = 0; i < input_shape.size(); ++i) {
            output_shape[i] = input_shape[i];
        }

        set_output_type(0, get_input_element_type(0), ngraph::PartialShape(output_shape));
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<FakeTestOp>(new_args.at(0));
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        return true;
    }
};

constexpr ngraph::NodeTypeInfo FakeTestOp::type_info;

class NewTestExtension : public InferenceEngine::IExtension {
public:
    NewTestExtension() {
        impls["Fake"] = [](const std::shared_ptr<ngraph::Node>& node) -> InferenceEngine::ILayerImpl::Ptr {
            return std::make_shared<NewFakePrimitiveImpl>(node);
        };
    }
    void Release() noexcept override { delete this; }

    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {
        static const InferenceEngine::Version VERSION{{}, "", ""};
        versionInfo = &VERSION;
    }

    void Unload() noexcept override {}

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        if (impls.find(node->description()) == impls.end())
            return {};
        return {"CPU"};
    }

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (impls.find(node->description()) == impls.end() || implType != "CPU")
            return nullptr;
        return impls[node->description()](node);
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<FakeTestOp>();
            opsets["experimental"] = opset;
        }
        return opsets;
    }
private:
    std::map<std::string, std::function<InferenceEngine::ILayerImpl::Ptr(const std::shared_ptr<ngraph::Node>)>> impls;
};

class smoke_ExtensionTest : public TestsCommon,
                            public TestsCommonFunc {

protected:
    void checkExtensionRemoved(extension_params p) {
        try {
            std::unique_ptr<InferenceEnginePluginPtr> score_engine;
            score_engine.reset(new InferenceEnginePluginPtr(make_plugin_name(p.plugin()).c_str()));
            (*score_engine)->SetConfig(p.config);
            ASSERT_EQ(p.extension.use_count(), 2);

            (*score_engine)->AddExtension(p.extension);
            // multi-device holds additional reference of the extension ptr
            ASSERT_EQ(p.extension.use_count(), p.pluginName.find("Multi")==std::string::npos ? 3 : 4);
            score_engine.reset();

            ASSERT_EQ(p.extension.use_count(), 2);
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }
    void checkExtensionNotRemovedFromAnotherEngineObject(extension_params p) {
        try {
            std::unique_ptr<InferenceEnginePluginPtr> score_engine1;
            score_engine1.reset(new InferenceEnginePluginPtr(make_plugin_name(p.plugin()).c_str()));
            (*score_engine1)->SetConfig(p.config);
            
            std::unique_ptr<InferenceEnginePluginPtr> score_engine2;
            score_engine2.reset(new InferenceEnginePluginPtr(make_plugin_name(p.plugin()).c_str()));
            (*score_engine2)->SetConfig(p.config);
            ASSERT_EQ(p.extension.use_count(), 2);

            (*score_engine1)->AddExtension(p.extension);
            // multi-device holds additional reference of the extension ptr
            ASSERT_EQ(p.extension.use_count(), p.pluginName.find("Multi")==std::string::npos ? 3 : 4);
            score_engine2.reset();

            // multi-device holds additional reference of the extension ptr
            ASSERT_EQ(p.extension.use_count(), p.pluginName.find("Multi")==std::string::npos ? 3 : 4);
            score_engine1.reset();
            ASSERT_EQ(p.extension.use_count(), 2);
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }

    void checkNotSharedExtensions(std::shared_ptr<IExtension> extension, std::string device) {
            std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="10" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Parameter" precision="FP32" version="opset1" id="0">
                    <data element_type="f32" shape="1,3,5,5"/>
                    <output>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="fake_layer" id="1" type="Fake" version="experimental" precision="FP32">
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" precision="FP32">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="output" type="Result" id="2" version="opset1">
                    <input>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
                <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
            </edges>
        </Net>
        )V0G0N";

        try {
            Core ie;
            ie.AddExtension(extension, "CPU");
            Core ie2;

            Blob::Ptr weights;
            CNNNetwork cnnNet1 = ie.ReadNetwork(model, weights);
            CNNNetwork cnnNet2 = ie2.ReadNetwork(model, weights);
            ASSERT_NO_THROW(ie.LoadNetwork(cnnNet1, device));
            ASSERT_THROW(ie2.LoadNetwork(cnnNet2, device), details::InferenceEngineException);
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }
};

/*************************************************
 * !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
 * All ref values was obtained from Caffe scoring
 * !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
 *************************************************/
#ifndef ENABLE_MKL_DNN
 #include "disable_tests.hpp"
#endif

TEST_F(smoke_ExtensionTest, MKLDNN_delete_extension) {
    std::shared_ptr<IExtension> ext(new NewTestExtension());
    checkExtensionRemoved({"MKLDNN", ext});
}

TEST_F(smoke_ExtensionTest, MKLDNN_no_delete_extension_from_another_engine) {
    std::shared_ptr<IExtension> ext(new NewTestExtension());
    checkExtensionNotRemovedFromAnotherEngineObject({"MKLDNN", ext});
}

TEST_F(smoke_ExtensionTest, MKLDNN_no_share_extension_between_engines) {
    std::shared_ptr<IExtension> ext(new NewTestExtension());
    checkNotSharedExtensions(ext, "CPU");
}

TEST_F(smoke_ExtensionTest, MKLDNN_no_share_new_extension_between_engines) {
    std::shared_ptr<IExtension> ext(new NewTestExtension());
    checkNotSharedExtensions(ext, "CPU");
}
