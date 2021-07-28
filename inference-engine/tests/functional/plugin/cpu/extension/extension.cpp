// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>
#include <onnx_import/onnx_utils.hpp>
#include <file_utils.h>
#include <common_test_utils/test_assertions.hpp>
#include <onnx_custom_op.hpp>


class CustomAbsKernel : public InferenceEngine::ILayerExecImpl {
public:
    explicit CustomAbsKernel(const std::shared_ptr<ngraph::Node>& node): node(node) {}

    InferenceEngine::StatusCode
    init(InferenceEngine::LayerConfig& /*config*/, InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                            InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
        InferenceEngine::LayerConfig layerConfig;
        layerConfig.dynBatchSupport = true;

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
        for (size_t i = 0; i < shape.size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                shape, {shape, order});
        layerConfig.outConfs.push_back(cfg);
        layerConfig.inConfs.push_back(cfg);
        conf.push_back(layerConfig);
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode
    execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
            InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
        for (size_t i = 0; i < inputs.size(); i++) {
            InferenceEngine::MemoryBlob::CPtr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputs[i]);
            InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputs[i]);
            if (!moutput || !minput) {
                return InferenceEngine::StatusCode::PARAMETER_MISMATCH;
            }
            // locked memory holder should be alive all time while access to its buffer happens
            auto minputHolder = minput->rmap();
            auto moutputHolder = moutput->wmap();

            auto inputData = minputHolder.as<const float *>();
            auto outputData = moutputHolder.as<float  *>();
            for (size_t j = 0; j < minput->size(); j++) {
                outputData[j] = inputData[j] < 0 ? (-inputData[j] * 2) : inputData[j];
            }
        }
        return InferenceEngine::StatusCode::OK;
    }

private:
    const std::shared_ptr<ngraph::Node> node;
};

class CustomAbs : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"CustomAbs", 100500};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }
    CustomAbs() = default;
    CustomAbs(const ngraph::Output<ngraph::Node>& arg): ngraph::op::Op({arg}) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        return std::make_shared<CustomAbs>(new_args.at(0));
    }
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        return true;
    }
};

constexpr ngraph::NodeTypeInfo CustomAbs::type_info;

class CustomAbsExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        std::map<std::string, ngraph::OpSet> opsets;
        ngraph::OpSet opset;
        opset.insert<CustomAbs>();
        opsets["custom_opset"] = opset;
        return opsets;
    }

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        if (node->description() != CustomAbs::type_info.name)
            return {};
        return {"CPU"};
    }

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        return std::make_shared<CustomAbsKernel>(node);
    }
};

void infer_model(InferenceEngine::Core& ie, const std::string& model,
                 const std::vector<float>& input_values, const std::vector<float>& expected) {
    InferenceEngine::Blob::CPtr weights;
    auto network = ie.ReadNetwork(model, weights);
    auto function = network.getFunction();

    auto network_inputs = network.getInputsInfo();
    auto network_outputs = network.getOutputsInfo();
    auto exe_network = ie.LoadNetwork(network, "CPU");
    auto inference_req = exe_network.CreateInferRequest();
    const auto& input = network_inputs.begin();
    const auto& input_info = input->second;

    auto blob = std::make_shared<InferenceEngine::TBlob<float>>(input_info->getTensorDesc());
    blob->allocate();
    ASSERT_EQ(input_values.size(), blob->size());
    float* blob_buffer = blob->wmap().template as<float*>();
    std::copy(input_values.begin(), input_values.end(), blob_buffer);
    inference_req.SetBlob(input->first, blob);

    inference_req.Infer();

    auto output = network_outputs.begin();
    InferenceEngine::MemoryBlob::CPtr computed = InferenceEngine::as<InferenceEngine::MemoryBlob>(inference_req.GetBlob(output->first));
    const auto computed_data = computed->rmap();
    const auto* computed_data_buffer = computed_data.template as<const float*>();
    std::vector<float> computed_values(computed_data_buffer,
                                   computed_data_buffer + computed->size());
    ASSERT_EQ(expected, computed_values);
}


TEST(Extension, OnnxModelWithCustomAbs) {
    std::string model = R"V0G0N(
ir_version: 3
producer_name: "nGraph ONNX Importer"
graph {
  node {
    input: "A"
    output: "Y"
    name: "customrelu"
    op_type: "CustomAbs"
    domain: "custom_domain"
  }
  name: "test_graph"
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 10
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 10
          }
        }
      }
    }
  }
}
opset_import {
  version: 1
  domain: "custom_domain"
}
)V0G0N";

    std::vector<float> input_values{1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    std::vector<float> expected{1, 4, 3, 8, 5, 12, 7, 16, 9, 20};
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<CustomAbsExtension>());
    ngraph::onnx_import::register_operator(
        CustomAbs::type_info.name, 1, "custom_domain", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
            ngraph::OutputVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<CustomAbs>(ng_inputs.at(0))};
    });

    infer_model(ie, model, input_values, expected);
    ngraph::onnx_import::unregister_operator(CustomAbs::type_info.name, 1, "custom_domain");
}


TEST(Extension, XmlModelWithCustomAbs) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="10"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="CustomAbs" version="custom_opset">
            <input>
                <port id="1" precision="FP32">
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    std::vector<float> expected{1, 4, 3, 8, 5, 12, 7, 16, 9, 20};
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<CustomAbsExtension>());
    infer_model(ie, model, input_values, expected);
}


static std::string get_extension_path() {
    return FileUtils::makePluginLibraryName<char>({}, std::string("template_extension") + IE_BUILD_POSTFIX);
}


TEST(Extension, XmlModelWithExtensionFromDSO) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="2,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="operation" id="1" type="Template" version="custom_opset">
            <data  add="11"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{12, 13, 14, 15, 16, 17, 18, 19};
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InferenceEngine::Extension>(get_extension_path()));
    infer_model(ie, model, input_values, expected);
}


TEST(Extension, OnnxModelWithExtensionFromDSO) {
    std::string model = R"V0G0N(
ir_version: 3
producer_name: "nGraph ONNX Importer"
graph {
  node {
    input: "A"
    output: "Y"
    name: "operation"
    op_type: "Template"
    domain: "custom_domain"
    attribute {
        name: "add"
        type: INT
        i: 11
    }
  }
  name: "test_graph"
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
}
opset_import {
  version: 1
  domain: "com.example"
}
)V0G0N";

    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{12, 13, 14, 15, 16, 17, 18, 19};
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InferenceEngine::Extension>(get_extension_path()));
    infer_model(ie, model, input_values, expected);
}


TEST(Extension, OnnxModelWithCustomReluDocsExample) {
    std::vector<float> input_values{0, -1, 2, -3, 4, -5, 6, -7};
    std::vector<float> expected{0, -3, 4, -9, 8, -15, 12, -21};

    register_custom_relu_operator();
    InferenceEngine::Core ie;
    infer_model(ie, custom_relu_model(), input_values, expected);
    unregister_custom_relu_operator();
}
