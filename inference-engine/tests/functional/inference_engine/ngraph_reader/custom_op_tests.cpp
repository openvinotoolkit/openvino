// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <ngraph/ngraph.hpp>
#include "ngraph_reader_tests.hpp"

class CustomAddConst : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"CustomAddConst", 100600};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }
    CustomAddConst() = default;
    CustomAddConst(const ngraph::Output<ngraph::Node>& arg, const ngraph::element::Type element_type,
        const ngraph::Shape shape, const std::shared_ptr<ngraph::runtime::AlignedBuffer> data):
        ngraph::op::Op({arg}),
        m_element_type(element_type),
        m_shape(shape),
        m_data(data) {
            constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, m_element_type, m_shape);
    }
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        return std::make_shared<CustomAddConst>(new_args.at(0), m_element_type, m_shape, m_data);
    }
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        visitor.on_attribute("element_type", m_element_type);
        visitor.on_attribute("shape", m_shape);
        if (!m_data) {
            m_data = std::make_shared<ngraph::runtime::AlignedBuffer>(shape_size(m_shape) * m_element_type.size(), 64);
        }
        visitor.on_attribute("value", m_data);
        return true;
    }

    ngraph::Shape getShapeAttr() const { return m_shape; }
    void* getDataPtr() { return (m_data ? m_data->get_ptr() : nullptr); }

    private:
        ngraph::element::Type m_element_type;
        ngraph::Shape m_shape{};
        std::shared_ptr<ngraph::runtime::AlignedBuffer> m_data;
};

constexpr ngraph::NodeTypeInfo CustomAddConst::type_info;

class CustomAddConstKernel : public InferenceEngine::ILayerExecImpl {
    public:
        explicit CustomAddConstKernel(const std::shared_ptr<ngraph::Node>& node): node(node) {
        try {
            auto castedNode = std::dynamic_pointer_cast<CustomAddConst>(node);
            if (!castedNode)
                THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
            if (castedNode->get_input_element_type(0) != ngraph::element::i32 || castedNode->get_output_element_type(0) != ngraph::element::i32)
                THROW_IE_EXCEPTION << "Operation supports only I32 tensors.";
            shape = castedNode->getShapeAttr();
            data_ptr = castedNode->getDataPtr();
            el_type = castedNode->get_input_element_type(0);
        } catch (InferenceEngine::details::InferenceEngineException& ex) {
            error = ex.what();
            }
        }

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
            cfg.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::I32,
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

                // retrieve constant data
                const auto data_beg = static_cast<char*>(data_ptr);
                const auto data_end = std::next(data_beg, shape_size(shape) * el_type.size());
                std::vector<char> value{data_beg, data_end};

                auto inputData = minputHolder.as<const int32_t *>();
                auto outputData = moutputHolder.as<int32_t  *>();
                for (size_t j = 0; j < minput->size(); j++) {
                    auto position = j * el_type.size() + 3;
                    outputData[j] = inputData[j] + static_cast<int32_t>(value.at(position));
                }
            }
            return InferenceEngine::StatusCode::OK;
        }

    private:
        const std::shared_ptr<ngraph::Node> node;
        ngraph::Shape shape;
        void* data_ptr;
        ngraph::element::Type el_type;
        std::string error;
};

class CustomAddConstExtension : public InferenceEngine::IExtension {
    public:
        CustomAddConstExtension() {
        }

        void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

        void Release() noexcept override { delete this; }

        void Unload() noexcept override {}

        std::map<std::string, ngraph::OpSet> getOpSets() override {
            std::map<std::string, ngraph::OpSet> opsets;
            ngraph::OpSet opset;
            opset.insert<CustomAddConst>();
            opsets["custom_opset"] = opset;
            return opsets;
        }

        std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
            if (node->description() != CustomAddConst::type_info.name)
                return {};
            return {"CPU"};
        }

        InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
            return std::make_shared<CustomAddConstKernel>(node);
        }
};

void infermodel(InferenceEngine::Core& ie, const std::string& model, const std::vector<int32_t>& input_values, const std::vector<int32_t>& expected) {
    InferenceEngine::Blob::CPtr weights;
    auto network = ie.ReadNetwork(model, weights);
    auto function = network.getFunction();

    auto network_inputs = network.getInputsInfo();
    auto network_outputs = network.getOutputsInfo();
    auto exe_network = ie.LoadNetwork(network, "CPU");
    auto inference_req = exe_network.CreateInferRequest();
    const auto& input = network_inputs.begin();
    const auto& input_info = input->second;

    auto blob = std::make_shared<InferenceEngine::TBlob<int32_t>>(input_info->getTensorDesc());
    blob->allocate();
    ASSERT_EQ(input_values.size(), blob->size());
    int32_t* blob_buffer = blob->wmap().template as<int32_t*>();
    std::copy(input_values.begin(), input_values.end(), blob_buffer);
    inference_req.SetBlob(input->first, blob);

    inference_req.Infer();

    auto output = network_outputs.begin();
    InferenceEngine::MemoryBlob::CPtr computed = InferenceEngine::as<InferenceEngine::MemoryBlob>(inference_req.GetBlob(output->first));
    const auto computed_data = computed->rmap();
    const auto* computed_data_buffer = computed_data.template as<const int32_t*>();
    std::vector<int32_t> computed_values(computed_data_buffer,
                                   computed_data_buffer + computed->size());
    ASSERT_EQ(expected, computed_values);
}

TEST_F(NGraphReaderTests, ReadXmlModelWithCustomAddConstOp) {
    std::string model = R"V0G0N(
  <net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
        <data element_type="i32" shape="4"/>
            <output>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="CustomAddConst" version="custom_opset">
        <data element_type="i32" shape="4" value="...%.../...5...>"/>
            <input>
                <port id="1" precision="I32">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="I32">
                    <dim>4</dim>
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

    std::vector<int32_t> input_values{2, 3, 4, 5};
    std::vector<int32_t> expected{39, 50, 57, 67};
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<CustomAddConstExtension>());
    infermodel(ie, model, input_values, expected);
}
