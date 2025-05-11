// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/util/log.hpp"
#include "tensor_lite_place.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace {
std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> decode_tensor_place(
    const ov::frontend::tensorflow_lite::TensorMetaInfo& tensor_meta_info,
    const ov::frontend::InputModel& model) {
    auto tensor_place = std::make_shared<ov::frontend::tensorflow_lite::TensorLitePlace>(
        model,
        tensor_meta_info.m_partial_shape,
        tensor_meta_info.m_element_type,
        std::vector<std::string>{tensor_meta_info.m_tensor_name},
        tensor_meta_info.m_quantization_info,
        tensor_meta_info.m_sparsity_info,
        tensor_meta_info.m_tensor_data);
    return tensor_place;
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> decode_input_tensor(
    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBaseOperation>& decoder,
    size_t idx,
    const ov::frontend::InputModel& model) {
    const auto& tensor_meta_info = decoder->get_input_tensor_info(idx);
    return decode_tensor_place(tensor_meta_info, model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> decode_output_tensor(
    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBaseOperation>& decoder,
    size_t idx,
    const ov::frontend::InputModel& model) {
    const auto& tensor_meta_info = decoder->get_output_tensor_info(idx);
    return decode_tensor_place(tensor_meta_info, model);
}
}  // namespace

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class InputModel::InputModelTFLiteImpl {
public:
    InputModelTFLiteImpl(const GraphIterator::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
    InputModelTFLiteImpl(const GraphIterator::Ptr& graph_iterator,
                         const ov::frontend::InputModel& input_model,
                         const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<ov::frontend::Place::Ptr> get_inputs() const;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const;
    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const;

    /////  Searching for places  /////
    std::vector<std::shared_ptr<OpPlace>> get_op_places() const {
        return m_op_places;
    }
    std::map<std::string, std::shared_ptr<TensorLitePlace>> get_tensor_places() const {
        return m_tensor_places;
    }
    std::map<std::string, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    }

    ///// Naming and annotation  /////
    void set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name);
    void add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name);
    void set_name_for_operation(const Place::Ptr& operation, const std::string& new_name);

    ///// Setting / getting tensor properties  /////
    void set_partial_shape(ov::frontend::Place::Ptr place, const ov::PartialShape& shape);
    ov::PartialShape get_partial_shape(ov::frontend::Place::Ptr place) const;
    void set_element_type(ov::frontend::Place::Ptr place, const ov::element::Type& type);
    ov::element::Type get_element_type(ov::frontend::Place::Ptr place) const;
    void set_tensor_value(ov::frontend::Place::Ptr place, const void* value);

    ///// Topology Editing  /////
    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                          const std::vector<ov::frontend::Place::Ptr>& outputs);

    std::vector<std::shared_ptr<ov::frontend::tensorflow_lite::InputModel>> get_subgraphs();

private:
    void load_model();
    void clean_up();

    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
    std::map<std::string, std::shared_ptr<TensorLitePlace>> m_tensor_places;
    std::vector<ov::frontend::Place::Ptr> m_inputs;
    std::vector<ov::frontend::Place::Ptr> m_outputs;
    std::map<std::string, Output<Node>> m_tensor_values;

    std::shared_ptr<GraphIterator> m_graph_iterator;
    const ov::frontend::InputModel& m_input_model;
    std::vector<std::shared_ptr<ov::frontend::tensorflow_lite::InputModel>> m_subgraphs;
    std::shared_ptr<TelemetryExtension> m_telemetry;
};

void InputModel::InputModelTFLiteImpl::load_model() {
    std::map<std::string, uint64_t> op_statistics;  // for telemetry

    m_op_places.reserve(m_graph_iterator->size());
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        const auto& decoder = m_graph_iterator->get_decoder();

        if (auto tensor_decoder = std::dynamic_pointer_cast<DecoderBaseTensor>(decoder)) {
            auto tensor_place = decode_tensor_place(tensor_decoder->get_tensor_info(), m_input_model);
            tensor_place->set_input_index(tensor_decoder->get_input_idx());
            tensor_place->set_output_index(tensor_decoder->get_output_idx());
            FRONT_END_GENERAL_CHECK(tensor_place->is_input() || tensor_place->is_output());
            auto name = tensor_place->get_names()[0];
            if (m_tensor_places.count(name) == 0) {
                m_tensor_places[name] = tensor_place;
                if (tensor_place->is_input())
                    m_inputs.push_back(tensor_place);
                if (tensor_place->is_output())
                    m_outputs.push_back(tensor_place);
            }
            continue;
        }
        m_op_places.push_back(std::make_shared<OpPlace>(m_input_model, decoder));

        if (m_telemetry) {
            op_statistics[decoder->get_op_type()]++;
        }

        auto operation_decoder = std::dynamic_pointer_cast<DecoderBaseOperation>(decoder);
        FRONT_END_GENERAL_CHECK(operation_decoder, "Operation decoder is expected");
        for (size_t i = 0; i < operation_decoder->get_input_size(); ++i) {
            auto place = decode_input_tensor(operation_decoder, i, m_input_model);
            auto name = place->get_names()[0];
            if (m_tensor_places.count(name) == 0) {
                m_tensor_places[name] = place;
                if (auto data = place->get_data()) {
                    auto constant = ov::op::v0::Constant::create(place->get_element_type(),
                                                                 place->get_partial_shape().to_shape(),
                                                                 data);
                    constant->set_friendly_name(name);
                    m_tensor_values[name] = constant;
                } else if (place->get_partial_shape() == PartialShape{0}) {  // empty constant
                    auto constant = ov::op::v0::Constant::create(place->get_element_type(),
                                                                 place->get_partial_shape().to_shape(),
                                                                 {});
                    constant->set_friendly_name(name);
                    m_tensor_values[name] = constant;
                } else {
                    FRONT_END_GENERAL_CHECK(false,
                                            "This tensor should be either input, constant or ",
                                            "should be already produced by previous operators: ",
                                            name,
                                            ". Error is encountered while working with operation of type ",
                                            operation_decoder->get_op_type(),
                                            " and name ",
                                            operation_decoder->get_op_name(),
                                            ".");
                }
            }
        }
        for (size_t i = 0; i < operation_decoder->get_output_size(); ++i) {
            auto place = decode_output_tensor(operation_decoder, i, m_input_model);
            auto name = place->get_names()[0];
            if (m_tensor_places.count(name) == 0)
                m_tensor_places[name] = place;
        }
    }

    auto sorting_places_by_idx = [](bool are_input_places) {
        return
            [are_input_places](const ov::frontend::Place::Ptr& lhs_place, const ov::frontend::Place::Ptr& rhs_place) {
                auto tflite_lhs_place =
                    std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(lhs_place);
                auto tflite_rhs_place =
                    std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(rhs_place);
                FRONT_END_GENERAL_CHECK(tflite_lhs_place != nullptr && tflite_rhs_place != nullptr,
                                        "TFLite Frontend works with TensorLitePlaces only");
                size_t rhs_idx, lhs_idx;
                if (are_input_places) {
                    lhs_idx = tflite_lhs_place->get_input_index();
                    rhs_idx = tflite_rhs_place->get_input_index();
                } else {
                    lhs_idx = tflite_lhs_place->get_output_index();
                    rhs_idx = tflite_rhs_place->get_output_index();
                }
                return lhs_idx < rhs_idx;
            };
    };
    std::sort(m_inputs.begin(), m_inputs.end(), sorting_places_by_idx(true));
    std::sort(m_outputs.begin(), m_outputs.end(), sorting_places_by_idx(false));

    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "tflite_" + op.first, static_cast<int>(op.second));
        }
    }

    size_t subgraph_size = m_graph_iterator->get_subgraph_size();
    if (subgraph_size > 1) {
        m_subgraphs.reserve(subgraph_size);
        m_subgraphs.push_back(nullptr);  // no main graph
        for (size_t i = 1; i < subgraph_size; ++i) {
            m_subgraphs.push_back(
                std::make_shared<ov::frontend::tensorflow_lite::InputModel>(m_graph_iterator->get_subgraph(i),
                                                                            m_telemetry));
        }
    }
}

InputModel::InputModelTFLiteImpl::InputModelTFLiteImpl(const GraphIterator::Ptr& graph_iterator,
                                                       const ov::frontend::InputModel& input_model)
    : m_graph_iterator(graph_iterator),
      m_input_model(input_model) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    load_model();
}

InputModel::InputModelTFLiteImpl::InputModelTFLiteImpl(const GraphIterator::Ptr& graph_iterator,
                                                       const ov::frontend::InputModel& input_model,
                                                       const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_graph_iterator(graph_iterator),
      m_input_model(input_model),
      m_telemetry(telemetry) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    load_model();
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFLiteImpl::get_inputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFLiteImpl::get_outputs() const {
    return m_outputs;
}

std::shared_ptr<TensorPlace> castToTensorPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlace>(place)) {
        return var_place;
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlace.");
}

ov::frontend::Place::Ptr InputModel::InputModelTFLiteImpl::get_place_by_tensor_name(
    const std::string& tensorName) const {
    if (m_tensor_places.find(tensorName) != m_tensor_places.end())
        return castToTensorPlace(m_tensor_places.at(tensorName));
    else
        return nullptr;
}

std::shared_ptr<OpPlace> castToOpPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<OpPlace>(place)) {
        return var_place;
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlace.");
}

void InputModel::InputModelTFLiteImpl::set_partial_shape(ov::frontend::Place::Ptr place, const PartialShape& shape) {
    castToTensorPlace(place)->set_partial_shape(shape);
}

ov::PartialShape InputModel::InputModelTFLiteImpl::get_partial_shape(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void InputModel::InputModelTFLiteImpl::set_element_type(ov::frontend::Place::Ptr place, const element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

ov::element::Type InputModel::InputModelTFLiteImpl::get_element_type(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::InputModelTFLiteImpl::set_tensor_value(ov::frontend::Place::Ptr place, const void* value) {
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    FRONT_END_GENERAL_CHECK(tensor_place->get_names().size() > 0,
                            "TensorFlow Lite Frontend: place to be frozen must have the name.");
    auto name = tensor_place->get_names()[0];
    FRONT_END_GENERAL_CHECK(p_shape.is_static(),
                            "TensorFlow Lite Frontend: specify static shape for " + name + " to be frozen.");
    FRONT_END_GENERAL_CHECK(type.is_static(),
                            "TensorFlow Lite Frontend: define static size type for " + name + " to be frozen.");
    auto constant = opset10::Constant::create(type, p_shape.to_shape(), value);
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

void InputModel::InputModelTFLiteImpl::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    castToTensorPlace(tensor)->set_names({new_name});
}

void InputModel::InputModelTFLiteImpl::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    auto tf_tensor = castToTensorPlace(tensor);
    auto names = tf_tensor->get_names();
    names.push_back(new_name);
    tf_tensor->set_names(names);
}

void InputModel::InputModelTFLiteImpl::set_name_for_operation(const Place::Ptr& operation,
                                                              const std::string& new_name) {
    auto op = castToOpPlace(operation);
    auto names = op->get_names();
    names.push_back(new_name);
    op->set_names(names);
}

void InputModel::InputModelTFLiteImpl::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    for (const auto& input_place : m_inputs) {
        auto input_lite_place = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(input_place);
        FRONT_END_GENERAL_CHECK(input_lite_place != nullptr, "Input Model has unexpected place as input");
        input_lite_place->set_input_index(-1);
    }
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
    clean_up();
}

void InputModel::InputModelTFLiteImpl::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    for (const auto& output_place : m_outputs) {
        auto output_lite_place =
            std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "Input Model has unexpected place as output");
        output_lite_place->set_output_index(-1);
    }
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
    clean_up();
}

void InputModel::InputModelTFLiteImpl::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                                        const std::vector<ov::frontend::Place::Ptr>& outputs) {
    for (const auto& output_place : m_outputs) {
        auto output_lite_place =
            std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "Input Model has unexpected place as output");
        output_lite_place->set_output_index(-1);
    }
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
    for (const auto& output_place : m_outputs) {
        auto output_lite_place =
            std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "Input Model has unexpected place as output");
        output_lite_place->set_output_index(-1);
    }
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
    clean_up();
}

void InputModel::InputModelTFLiteImpl::clean_up() {
    // TODO: remove all the unnecessary tensors and operations. Could be postponed as TF Lite is OOB type of FrontEnd
}

std::vector<std::shared_ptr<ov::frontend::tensorflow_lite::InputModel>>
InputModel::InputModelTFLiteImpl::get_subgraphs() {
    return m_subgraphs;
}

InputModel::InputModel(const GraphIterator::Ptr& graph_iterator, const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelTFLiteImpl>(graph_iterator, *this, telemetry)} {}

std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> InputModel::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace>> InputModel::get_tensor_places()
    const {
    return _impl->get_tensor_places();
}

std::map<std::string, Output<Node>> InputModel::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return _impl->get_inputs();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return _impl->get_outputs();
}

ov::frontend::Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->get_place_by_tensor_name(tensorName);
}

ov::frontend::Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    FRONT_END_NOT_IMPLEMENTED(get_place_by_input_index);
}

void InputModel::set_partial_shape(const Place::Ptr& place, const PartialShape& shape) {
    _impl->set_partial_shape(place, shape);
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    return _impl->get_partial_shape(place);
}

void InputModel::set_element_type(const Place::Ptr& place, const element::Type& type) {
    _impl->set_element_type(place, type);
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    return _impl->get_element_type(place);
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    _impl->set_tensor_value(place, value);
}

void InputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    _impl->set_name_for_tensor(tensor, new_name);
}

void InputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    _impl->add_name_for_tensor(tensor, new_name);
}

void InputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    _impl->set_name_for_operation(operation, new_name);
}

void InputModel::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->override_all_outputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    _impl->override_all_inputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                  const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->extract_subgraph(inputs, outputs);
}

std::vector<std::shared_ptr<InputModel>> InputModel::get_subgraphs() const {
    return _impl->get_subgraphs();
}

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
