// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <pugixml.hpp>

#include "ir_deserializer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "utils.hpp"

namespace {
void parse_pre_process(pugi::xml_node& root,
                       std::shared_ptr<ov::AlignedBuffer> weights,
                       std::shared_ptr<ov::Model> model) {
    /* Preprocessing block can have two preprocessing types:
     *
     * <pre-process mean-precision="FP32" reference-layer-name="data">
     *     <channel id="0">
     *         <mean value="1.1"/>
     *     </channel>
     * </pre-process>
     *
     * OR
     *
     * <pre-process mean-precision="FP32" reference-layer-name="data">
     *     <channel id="0">
     *         <mean offset="0" size="1936"/>
     *     </channel>
     * </pre-process>
     */

    auto ppNode = root.child("pre-process");
    if (ppNode.empty()) {
        return;
    }
    // find out to what input this belongs to
    std::string inputName;
    std::shared_ptr<ov::Node> input_node;

    inputName = ov::util::pugixml::get_str_attr(ppNode, "reference-layer-name", "");
    inputName = ov::util::trim(inputName);

    if (inputName.empty()) {
        // fallback (old format), look for the picture in the inputs
        for (const auto& parameter : model->get_parameters()) {
            if (parameter->get_partial_shape().rank().is_static() &&
                parameter->get_partial_shape().rank().get_length() == 4) {
                input_node = parameter;
                break;
            }
        }

        OPENVINO_ASSERT(!model->get_parameters().empty());
        if (!input_node) {
            input_node = model->get_parameters()[0];
        }

        inputName = input_node->get_friendly_name();
    } else {
        for (const auto& parameter : model->get_parameters()) {
            if (parameter->get_friendly_name() == inputName) {
                input_node = parameter;
                break;
            }
        }
    }

    OPENVINO_ASSERT(input_node, "pre-process name ref '", inputName, "' refers to un-existing input");

    const auto& input_shape = input_node->output(0).get_partial_shape();
    OPENVINO_ASSERT(!input_shape.is_dynamic(), "can not apply pre-process for '", inputName, "' input");

    ov::Shape mean_scalar_shape;  // [C, 1 ... 1]
    ov::Shape mean_shape;         // [1, H, W] - for 4D case

    const auto inputDims = input_shape.to_shape();

    OPENVINO_ASSERT(inputDims.size() >= 2, "network did not define input dimensions properly");

    if (inputDims.size() == 2) {  // NC
        mean_scalar_shape = {inputDims[1]};
        mean_shape = {1};
    } else if (inputDims.size() == 3) {  // CHW - legacy representation for 3D input shape
        mean_scalar_shape = {inputDims[0], 1, 1};
        mean_shape = {1, inputDims[1], inputDims[2]};
    } else if (inputDims.size() == 4) {  // NCHW
        mean_scalar_shape = {inputDims[1], 1, 1};
        mean_shape = {1, inputDims[2], inputDims[3]};
    } else if (inputDims.size() == 5) {  // NCDHW
        mean_scalar_shape = {inputDims[1], 1, 1, 1};
        mean_shape = {1, inputDims[2], inputDims[3], inputDims[4]};
    }
    const size_t channels = mean_scalar_shape[0];

    uint64_t next_channel_id{0};
    std::set<std::pair<uint64_t, float>> mean_scalar_values;
    std::set<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> mean_values;

    auto input_type = input_node->get_output_element_type(0);
    FOREACH_CHILD (chan, ppNode, "channel") {
        auto chanNo = ov::util::pugixml::get_uint64_attr(chan, "id", next_channel_id++);

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("value") && (!meanNode.attribute("size"))) {
                OPENVINO_THROW("mean should have at least one of the following attribute: value, size");
            }
            if (meanNode.attribute("value")) {
                mean_scalar_values.insert({chanNo, ov::util::pugixml::get_float_attr(meanNode, "value")});
            }
            if (meanNode.attribute("size") && meanNode.attribute("offset")) {
                auto const_size = ov::util::pugixml::get_uint64_attr(meanNode, "size");
                auto const_offset = ov::util::pugixml::get_uint64_attr(meanNode, "offset");
                if (shape_size(mean_shape) * input_type.size() != const_size) {
                    OPENVINO_THROW("mean blob size mismatch expected input, got: ",
                                   const_size,
                                   " expecting ",
                                   mean_shape,
                                   " x ",
                                   input_type.size());
                }
                if (const_offset + const_size > weights->size()) {
                    OPENVINO_THROW("mean value offset and size are out of weights size range");
                }
                mean_values.insert({chanNo, {const_size, const_offset}});
            }
        }
    }

    if (!mean_values.empty() && !mean_scalar_values.empty()) {
        OPENVINO_THROW("mean values have different types");
    }

    if (!mean_scalar_values.empty()) {
        if (mean_scalar_values.size() != channels) {
            OPENVINO_THROW("Number of mean values (",
                           mean_scalar_values.size(),
                           ") is not equal to number of channels (",
                           channels,
                           ")");
        }
        std::vector<float> values(channels);
        for (const auto& item : mean_scalar_values) {
            if (item.first >= channels) {
                OPENVINO_THROW("Mean values channel index ", item.first, " is out of range (", channels, ")");
            }
            values[item.first] = item.second;
        }
        auto mean_values_constant = ov::op::v0::Constant::create(input_type, mean_scalar_shape, values);

        const auto& consumers = input_node->output(0).get_target_inputs();
        auto add = std::make_shared<ov::op::v1::Subtract>(input_node, mean_values_constant);
        for (const auto& consumer : consumers) {
            consumer.replace_source_output(add);
        }
    }

    if (!mean_values.empty()) {
        if (mean_values.size() != channels) {
            OPENVINO_THROW("Number of mean values (",
                           mean_values.size(),
                           ") is not equal to number of channels (",
                           channels,
                           ")");
        }
        ov::NodeVector per_channel_values(channels);
        for (const auto& item : mean_values) {
            if (item.first >= channels) {
                OPENVINO_THROW("Mean values channel index ", item.first, " is out of range (", channels, ")");
            }
            const size_t offset = item.second.second;
            const char* data = weights->get_ptr<char>() + offset;
            per_channel_values[item.first] = ov::op::v0::Constant::create(input_type, mean_shape, data);
        }
        auto const_node =
            ov::util::get_constant_from_source(std::make_shared<ov::op::v0::Concat>(per_channel_values, 0));
        OPENVINO_ASSERT(const_node);
        const auto& consumers = input_node->output(0).get_target_inputs();
        auto add = std::make_shared<ov::op::v1::Subtract>(input_node, const_node);
        for (const auto& consumer : consumers) {
            consumer.replace_source_output(add);
        }
    }
}
}  // namespace

namespace ov {
namespace frontend {
namespace ir {

class InputModel::InputModelIRImpl {
    std::shared_ptr<ov::AlignedBuffer> m_weights;
    std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> m_extensions;
    std::unordered_map<std::string, ov::OpSet> m_opsets;
    pugi::xml_node m_root;
    pugi::xml_document m_xml_doc;
    std::string m_weights_path;

public:
    InputModelIRImpl(std::istream& model,
                     const std::shared_ptr<ov::AlignedBuffer>& weights,
                     const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                     std::string weights_path)
        : m_weights(weights),
          m_extensions(extensions),
          m_weights_path(std::move(weights_path)) {
        pugi::xml_parse_result res = m_xml_doc.load(model);
        OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
        init_opset();
    }

    InputModelIRImpl(const std::shared_ptr<ov::AlignedBuffer>& model,
                     const std::shared_ptr<ov::AlignedBuffer>& weights,
                     const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                     std::string weights_path)
        : m_weights(weights),
          m_extensions(extensions),
          m_weights_path(std::move(weights_path)) {
        auto res = m_xml_doc.load_buffer(model->get_ptr(), model->size(), pugi::parse_default, pugi::encoding_utf8);
        OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
        init_opset();
    }

    std::shared_ptr<ov::Model> convert();

private:
    void init_opset() {
        m_root = m_xml_doc.document_element();
        for (const auto& it : ov::get_available_opsets()) {
            m_opsets[it.first] = it.second();
        }
    }
};

InputModel::InputModel(std::istream& model,
                       const std::shared_ptr<ov::AlignedBuffer>& weights,
                       const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                       std::string weights_path) {
    _impl = std::make_shared<InputModelIRImpl>(model, weights, extensions, std::move(weights_path));
}

InputModel::InputModel(const std::shared_ptr<ov::AlignedBuffer>& model,
                       const std::shared_ptr<ov::AlignedBuffer>& weights,
                       const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                       std::string weights_path) {
    _impl = std::make_shared<InputModelIRImpl>(model, weights, extensions, std::move(weights_path));
}

std::shared_ptr<ov::Model> InputModel::convert() {
    return _impl->convert();
}

std::shared_ptr<ov::Model> InputModel::InputModelIRImpl::convert() {
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;

    // Load default opsets
    size_t version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(m_root, "version", 0));
    ov::XmlDeserializer visitor(m_root, m_weights, m_opsets, m_extensions, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);
    model->get_rt_info()["version"] = int64_t(version);
    if (!m_weights_path.empty())
        model->get_rt_info()["__weights_path"] = m_weights_path;
    parse_pre_process(m_root, m_weights, model);

    return model;
}

}  // namespace ir
}  // namespace frontend
}  // namespace ov
