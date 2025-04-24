// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <fstream>
#if defined(__MINGW32__) || defined(__MINGW64__)
#    include <filesystem>
#endif
#include <queue>

#include "decoder_proto.hpp"
#include "framework.pb.h"
#include "input_model.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "paddle_utils.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace paddle {

using namespace ::paddle::framework::proto;

class InputModel::InputModelImpl {
public:
    template <typename T>
    InputModelImpl(const std::basic_string<T>& path,
                   const InputModel& input_model,
                   const std::shared_ptr<TelemetryExtension>& telemetry);
    InputModelImpl(const std::vector<std::istream*>& streams,
                   const InputModel& input_model,
                   const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<Place::Ptr> get_inputs() const;
    std::vector<Place::Ptr> get_outputs() const;
    int64_t get_version() const {
        return m_fw_ptr->version().version();
    }
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs);
    void override_all_inputs(const std::vector<Place::Ptr>& inputs);
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);
    void set_default_shape(Place::Ptr place, const ov::Shape&);
    void set_partial_shape(Place::Ptr place, const ov::PartialShape&);
    ov::PartialShape get_partial_shape(Place::Ptr place) const;
    void set_element_type(Place::Ptr place, const ov::element::Type&);
    ov::element::Type get_element_type(const Place::Ptr& place) const;
    void set_tensor_value(Place::Ptr place, const void* value);
    std::vector<std::shared_ptr<OpPlace>> get_op_places(const int32_t blck_idx) const;
    std::map<std::string, std::shared_ptr<TensorPlace>> get_var_places() const {
        return m_var_places;
    }
    std::map<paddle::TensorName, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    };

private:
    void load_places();
    template <typename T>
    void load_consts(const std::basic_string<T>& folder_with_weights);
    void load_consts(std::istream* weight_stream);
    void create_temp_consts();
    std::vector<std::shared_ptr<OpPlace>> determine_cut_nodes() const;

    std::vector<std::vector<std::shared_ptr<OpPlace>>> m_op_places;
    std::map<std::string, std::shared_ptr<TensorPlace>> m_var_places;
    std::shared_ptr<ProgramDesc> m_fw_ptr;
    const InputModel& m_input_model;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::map<paddle::TensorName, Output<Node>> m_tensor_values;

    std::shared_ptr<TelemetryExtension> m_telemetry;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};

void InputModel::InputModelImpl::load_places() {
    const int cnt_of_blocks = m_fw_ptr->blocks_size();
    const auto& blocks = m_fw_ptr->blocks();
    std::map<std::string, uint64_t> op_statistics;

    m_op_places.resize(cnt_of_blocks);

    for (int block_idx = 0; block_idx < cnt_of_blocks; block_idx++) {
        const auto& block = blocks[block_idx];

        for (const auto& var : block.vars()) {
            m_var_places[var.name()] = std::make_shared<TensorPlace>(m_input_model, var);
        }

        for (const auto& op : block.ops()) {
            auto op_place = std::make_shared<OpPlace>(m_input_model, op);
            op_place->set_decoder(std::make_shared<DecoderProto>(op_place));

            if (m_telemetry) {
                op_statistics[op.type()]++;
            }

            m_op_places[block_idx].push_back(op_place);

            for (const auto& output : op.outputs()) {
                for (const auto& var_name : output.arguments()) {
                    auto out_port = std::make_shared<OutPortPlace>(m_input_model);

                    // connect out_port and tensor
                    const auto& tensor = m_var_places.at(var_name);
                    tensor->add_producing_port(out_port);
                    out_port->set_target_tensor(tensor);

                    // connect out_port and op
                    op_place->add_out_port(out_port, output.parameter());
                    out_port->set_op(op_place);
                }
            }

            for (const auto& input : op.inputs()) {
                for (const auto& var_name : input.arguments()) {
                    auto in_port = std::make_shared<InPortPlace>(m_input_model);

                    // connect in_port and tensor
                    const auto& tensor = m_var_places.at(var_name);
                    tensor->add_consuming_port(in_port);
                    in_port->set_source_tensor(tensor);

                    // connect in_port and op
                    op_place->add_in_port(in_port, input.parameter());
                    in_port->set_op(op_place);
                }
            }

            // Determine outputs and inputs
            if (op.type() == "feed") {
                const auto& place = op_place->get_output_port_paddle("Out", 0);
                const auto& var_place = std::dynamic_pointer_cast<TensorPlace>(place->get_target_tensor_paddle());
                const auto& tensor_desc = var_place->get_desc().type().lod_tensor().tensor();
                const auto& dims = tensor_desc.dims();

                var_place->set_element_type(get_ov_type(tensor_desc.data_type()));
                var_place->set_partial_shape(PartialShape(std::vector<Dimension>(dims.begin(), dims.end())));
                m_inputs.push_back(var_place);
            } else if (op.type() == "fetch") {
                auto place = op_place->get_input_port_paddle("X", 0);
                m_outputs.push_back(place->get_source_tensor_paddle());
            }
        }
    }
    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "paddle_" + op.first, static_cast<int>(op.second));
        }
    }
}

namespace {
bool read_tensor(std::istream& is, char* data, size_t len) {
    is.read(data, len);
    return (size_t)is.gcount() == len;
}

template <typename T>
std::basic_string<T> get_const_path(const std::basic_string<T>& folder_with_weights, const std::string& name) {
    return folder_with_weights + paddle::get_path_sep<T>() + name;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_const_path(const std::basic_string<wchar_t>& folder, const std::string& name) {
    return folder + paddle::get_path_sep<wchar_t>() + ov::util::string_to_wstring(name);
}
#endif

template <typename T>
bool is_pdmodel(const std::basic_string<T>& path) {
    std::string ext = ".pdmodel";
    return ov::util::ends_with(path, ext);
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
bool is_pdmodel(const std::basic_string<wchar_t>& path) {
    std::wstring ext = L".pdmodel";
    return ov::util::ends_with(path, ext);
}
#endif

template <typename T>
std::basic_string<T> get_model_path(const std::basic_string<T>& path, std::ifstream* weights_stream) {
    std::string model_file{path};
    std::string ext = ".pdmodel";
    if (ov::util::ends_with(model_file, ext)) {
        std::string params_ext = ".pdiparams";
        std::string weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file, std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += paddle::get_path_sep<T>() + "__model__";
    }
    return model_file;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_model_path(const std::basic_string<wchar_t>& path, std::ifstream* weights_stream) {
    std::wstring model_file{path};
    std::wstring ext = L".pdmodel";
    if (ov::util::ends_with(model_file, ext)) {
        std::wstring params_ext = L".pdiparams";
        std::wstring weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file.c_str(), std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += paddle::get_path_sep<wchar_t>() + L"__model__";
    }
    return model_file;
}
#endif
}  // namespace

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelImpl::get_op_places(const int32_t blck_idx) const {
    if (m_graph_changed) {
        return determine_cut_nodes();
    }
    if (static_cast<size_t>(blck_idx) < m_op_places.size())
        return m_op_places[blck_idx];
    return {};
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelImpl::determine_cut_nodes() const {
    std::queue<OpPlace*> q;
    std::unordered_set<OpPlace*> visited;
    std::vector<std::shared_ptr<OpPlace>> new_op_places;
    new_op_places.reserve(m_op_places[0].size());
    // Marking nodes from outputs to inputs/constants
    for (const auto& output : get_outputs()) {
        if (!output->is_input()) {
            auto paddle_output_op = std::dynamic_pointer_cast<OpPlace>(output->get_producing_operation());
            FRONT_END_GENERAL_CHECK(paddle_output_op != nullptr, "Output doesn't have producing operation");
            if (!visited.count(paddle_output_op.get())) {
                visited.insert(paddle_output_op.get());
                q.push(paddle_output_op.get());
                new_op_places.push_back(paddle_output_op);
            }
        }
    }
    while (!q.empty()) {
        auto p_op = q.front();
        q.pop();
        for (const auto& map_pair : p_op->get_input_ports()) {
            for (const auto& port : map_pair.second) {
                auto tensor = port->get_source_tensor();
                if (tensor && !tensor->is_input() && !m_tensor_values.count(tensor->get_names()[0])) {
                    std::shared_ptr<OpPlace> paddle_op =
                        std::dynamic_pointer_cast<OpPlace>(tensor->get_producing_operation());
                    if (paddle_op && !visited.count(paddle_op.get())) {
                        visited.insert(paddle_op.get());
                        q.push(paddle_op.get());
                        new_op_places.push_back(paddle_op);
                    }
                }
            }
        }
    }
    std::reverse(new_op_places.begin(), new_op_places.end());
    return new_op_places;
}

// load_consts with folder is compatible with old PaddlePaddle API.
template <typename T>
void InputModel::InputModelImpl::load_consts(const std::basic_string<T>& folder_with_weights) {
    for (const auto& item : m_var_places) {
        const auto& var_desc = item.second->get_desc();
        const auto& name = item.first;
        if (ov::util::ends_with(name, std::string{"feed"}) || ov::util::ends_with(name, std::string{"fetch"}))
            continue;
        if (!var_desc.persistable())
            continue;

        FRONT_END_GENERAL_CHECK(var_desc.type().type() == ::paddle::framework::proto::VarType::LOD_TENSOR);
        const auto& tensor = var_desc.type().lod_tensor().tensor();
        Shape shape(tensor.dims().cbegin(), tensor.dims().cend());
        const auto& type = get_ov_type(tensor.data_type());
        const auto& data_length = shape_size(shape) * type.size();
        std::vector<uint8_t> tensor_data(data_length);

        bool read_succeed = false;
        if (!folder_with_weights.empty()) {
#if defined(__MINGW32__) || defined(__MINGW64__)
            std::ifstream is(std::filesystem::path(get_const_path(folder_with_weights, name)),
                             std::ios::in | std::ifstream::binary);
#else
            std::ifstream is(get_const_path(folder_with_weights, name), std::ios::in | std::ifstream::binary);
#endif
            FRONT_END_GENERAL_CHECK(is && is.is_open(), "Cannot open file for constant value.");
            const size_t header_size = 16;
            std::vector<char> header(header_size);
            is.read(&header[0], header_size);

            uint32_t dims_len = 0;
            is.read(reinterpret_cast<char*>(&dims_len), 4);
            std::vector<char> dims_struct(dims_len);
            is.read(&dims_struct[0], dims_len);
            read_succeed = read_tensor(is, reinterpret_cast<char*>(&tensor_data[0]), data_length);
        } else {
            FRONT_END_GENERAL_CHECK(false, "Folder with weights must be provided.");
        }
        FRONT_END_GENERAL_CHECK(read_succeed,
                                "File containing constant with name ",
                                name,
                                " wasn't successfully read.");
        auto const_node = opset7::Constant::create(type, shape, &tensor_data[0]);
        const_node->set_friendly_name(name);
        m_tensor_values[name] = const_node;
    }
}

// load_consts with stream is compatible with new PaddlePaddle API.
void InputModel::InputModelImpl::load_consts(std::istream* weight_stream) {
    for (const auto& item : m_var_places) {
        const auto& var_desc = item.second->get_desc();
        const auto& name = item.first;
        if (ov::util::ends_with(name, std::string{"feed"}) || ov::util::ends_with(name, std::string{"fetch"}))
            continue;

        // var_desc.persistable() is used to mark node const value or not.
        if (!var_desc.persistable())
            continue;

        FRONT_END_GENERAL_CHECK(var_desc.type().type() == ::paddle::framework::proto::VarType::LOD_TENSOR);
        FRONT_END_GENERAL_CHECK(weight_stream != nullptr && weight_stream->peek() != EOF,
                                "PaddlePaddle *.pdiparams format weight file doesn't exist!");
        /*
            reference:
            https://github.com/PaddlePaddle/Paddle2ONNX/blob/c14446437041a0aa3572994d085b7a35c5b0985c/paddle2onnx/parser/parser.cc#L261
            When deserialize the proto, the header of each weight
            [ 4 byte ]      -- version(not need)
            [   8 byte   ]  -- lod_level(not need)
            [ 4 byte ]      -- version(not need)
            [ 4 byte ]      -- TensorDesc size
            [ x byte ... ]  -- TensorDesc
            [ y byte ... ]  -- weight
        */
        {
            const size_t header_size = 16;
            std::vector<char> header(header_size);
            weight_stream->read(&header[0], header_size);
        }

        int32_t size;
        weight_stream->read(reinterpret_cast<char*>(&size), sizeof(size));

        std::unique_ptr<char[]> buf(new char[size]);
        weight_stream->read(reinterpret_cast<char*>(buf.get()), size);

        std::unique_ptr<::paddle::framework::proto::VarType_TensorDesc> tensor_desc(
            new ::paddle::framework::proto::VarType_TensorDesc());
        tensor_desc->ParseFromArray(buf.get(), size);
        Shape shape(tensor_desc->dims().cbegin(), tensor_desc->dims().cend());
        const auto& type = get_ov_type(tensor_desc->data_type());
        const auto& data_length = shape_size(shape) * type.size();
        std::vector<uint8_t> tensor_data(data_length);

        bool read_succeed = read_tensor(*weight_stream, reinterpret_cast<char*>(&tensor_data[0]), data_length);
        FRONT_END_GENERAL_CHECK(read_succeed,
                                "File containing constant with name ",
                                name,
                                " wasn't successfully read.");

        auto const_node = opset7::Constant::create(type, shape, &tensor_data[0]);
        const_node->set_friendly_name(name);
        m_tensor_values[name] = const_node;
    }
}

/*
    1. path: is a directory, compatible with old PaddlePaddle API.
             read __model__ as model stream.
             read the separate weights in the directory.
    2. path: is a pdmodel file, compatible with new PaddlePaddle API.
             read *.pdmodel as model stream.
             read *.pdiparam as weight stream.
*/
template <typename T>
InputModel::InputModelImpl::InputModelImpl(const std::basic_string<T>& path,
                                           const InputModel& input_model,
                                           const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model),
      m_telemetry(telemetry) {
    std::ifstream weights_stream;
    std::ifstream pb_stream(get_model_path<T>(path, &weights_stream).c_str(), std::ios::in | std::ifstream::binary);

    FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(),
                            "Could not open the file: \"",
                            util::path_to_string(path),
                            '"');
    FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(&pb_stream), "Model can't be parsed");
    // According to Paddle, the saved model has the framework version
    // For example Paddle 2.1.0 is encoded as 2001000. 0 means the latest framework.
    // https://github.com/paddle/Paddle/blob/develop/cmake/version.cmake
    // https://github.com/paddle/Paddle/blob/2100816c5190693cc7dee181e96af72e9f0fbd1d/paddle/fluid/framework/program_desc.cc#L52
    int64_t version = m_fw_ptr->version().version();
    FRONT_END_GENERAL_CHECK(
        version >= 2000000 || version == 0,
        "[Frontend]Only Support Paddle greater than 2.0.0, current version " + std::to_string(version));
    load_places();
    if (is_pdmodel(path)) {
        load_consts(&weights_stream);
    } else {
        load_consts(path);
    }
    create_temp_consts();
}

void InputModel::InputModelImpl::create_temp_consts() {
    for (const auto& item : m_var_places) {
        const auto& var_place = item.second;
        const auto& var_desc = var_place->get_desc();
        const auto& name = item.first;
        if (var_desc.persistable())
            continue;

        // The node with tensorarray as its input may be created before the node with this tensorarray
        // as its output. e.g. the tensorarray is both the input and output of the same node.
        // So we have to create a fake empty node here.
        // Problem is, we have no idea which axis should be 0.
        // Since the models (faster/mask rcnn) are either concating tensors in tensorarray along the dynamic
        // dimension, or concating static shape tensors. So we make the dynamic dimension to be 0. In case of static
        // shape, we simply the the first dimension be 0.
        if (var_desc.type().has_tensor_array()) {
            const auto& tensor = var_desc.type().tensor_array().tensor();
            const auto& type = get_ov_type(tensor.data_type());

            std::cout << "WARNING: The PaddlePaddle model has \"TENSOR_ARRAY\" variables, which is supported "
                      << " under limited situations.\n";

            PartialShape tensor_ps(std::vector<Dimension>(tensor.dims().cbegin(), tensor.dims().cend()));
            tensor_ps.insert(tensor_ps.begin(), 1);  // unsqueeze
            // also update the place for following initialize the graph connection
            var_place->set_element_type(type);
            var_place->set_partial_shape(tensor_ps);

            Shape shape(tensor_ps.size(), 0);
            for (size_t i = 0; i < tensor_ps.size(); i++) {
                const auto& dim = tensor_ps[i];
                if (dim.is_static()) {
                    shape[i] = dim.get_length();
                }
            }

            if (tensor_ps.is_static()) {
                // this tensorarray tensor originally could be scalar, then
                // tensor_ps size would be 1 after unsqueeze.
                auto idx = tensor_ps.size() > 1 ? 1 : 0;
                shape[idx] = 0;
            }

            auto node = opset7::Constant::create(type, shape, {0});
            node->set_friendly_name(name);
            node->output(0).get_tensor().add_names({name});

            m_tensor_values[name] = node;
        }
    }
}

InputModel::InputModelImpl::InputModelImpl(const std::vector<std::istream*>& streams,
                                           const InputModel& input_model,
                                           const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model),
      m_telemetry(telemetry) {
    if (streams.size() != 1) {
        FRONT_END_GENERAL_CHECK(streams.size() == 2,
                                "Two streams are needed to load a model: model and weights streams");
    }
    FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(streams[0]), "Model can't be parsed");
    int64_t version = m_fw_ptr->version().version();
    FRONT_END_GENERAL_CHECK(
        version >= 2000000 || version == 0,
        "[Frontend]Only Support Paddle greater than 2.0.0, current version " + std::to_string(version));
    load_places();
    if (streams.size() > 1)
        load_consts(streams[1]);
    create_temp_consts();
}

std::vector<Place::Ptr> InputModel::InputModelImpl::get_inputs() const {
    return m_inputs;
}

std::vector<Place::Ptr> InputModel::InputModelImpl::get_outputs() const {
    return m_outputs;
}

Place::Ptr InputModel::InputModelImpl::get_place_by_tensor_name(const std::string& tensorName) const {
    if (m_var_places.count(tensorName))
        return m_var_places.at(tensorName);
    return nullptr;
}

namespace {
std::shared_ptr<TensorPlace> castToTensorPlace(const Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlace>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlace>(place)) {
        return in_port_place->get_source_tensor_paddle();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlace>(place)) {
        return out_port_place->get_target_tensor_paddle();
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlacepaddle.");
}

}  // namespace

void InputModel::InputModelImpl::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    m_graph_changed = true;
    m_inputs.clear();
    for (const auto& inp : inputs) {
        m_inputs.push_back(castToTensorPlace(inp));
    }
}

void InputModel::InputModelImpl::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    m_outputs.clear();
    for (const auto& outp : outputs) {
        m_outputs.push_back(castToTensorPlace(outp));
    }
}

void InputModel::InputModelImpl::extract_subgraph(const std::vector<Place::Ptr>& inputs,
                                                  const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    override_all_inputs(inputs);
    override_all_outputs(outputs);
}

void InputModel::InputModelImpl::set_default_shape(Place::Ptr place, const ov::Shape& shape) {
    FRONT_END_NOT_IMPLEMENTED("set_default_shape");
}

void InputModel::InputModelImpl::set_partial_shape(Place::Ptr place, const ov::PartialShape& p_shape) {
    castToTensorPlace(place)->set_partial_shape(p_shape);
}

ov::PartialShape InputModel::InputModelImpl::get_partial_shape(Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void InputModel::InputModelImpl::set_element_type(Place::Ptr place, const ov::element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

ov::element::Type InputModel::InputModelImpl::get_element_type(const Place::Ptr& place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::InputModelImpl::set_tensor_value(Place::Ptr place, const void* value) {
    m_graph_changed = true;
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->get_names()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

InputModel::InputModel(const std::string& path, const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelImpl>(path, *this, telemetry)} {}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModel::InputModel(const std::wstring& path, const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelImpl>(path, *this, telemetry)} {}
#endif

InputModel::InputModel(const std::vector<std::istream*>& streams, const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelImpl>(streams, *this, telemetry)} {}

std::vector<std::shared_ptr<OpPlace>> InputModel::get_op_places(const int32_t blck_idx) const {
    return _impl->get_op_places(blck_idx);
}

std::map<std::string, std::shared_ptr<TensorPlace>> InputModel::get_var_places() const {
    return _impl->get_var_places();
}

std::map<paddle::TensorName, Output<Node>> InputModel::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<Place::Ptr> InputModel::get_inputs() const {
    return _impl->get_inputs();
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    return _impl->get_outputs();
}

int64_t InputModel::get_version() const {
    return _impl->get_version();
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->get_place_by_tensor_name(tensorName);
}

Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    FRONT_END_NOT_IMPLEMENTED(get_place_by_input_index);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    _impl->override_all_outputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    _impl->override_all_inputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    _impl->extract_subgraph(inputs, outputs);
}

void InputModel::set_partial_shape(const Place::Ptr& place, const ov::PartialShape& p_shape) {
    _impl->set_partial_shape(place, p_shape);
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    return _impl->get_partial_shape(place);
}

void InputModel::set_element_type(const Place::Ptr& place, const ov::element::Type& type) {
    _impl->set_element_type(place, type);
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    _impl->set_tensor_value(place, value);
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
