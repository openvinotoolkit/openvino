// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <editor.hpp>
#include <memory>
#include <openvino/frontend/place.hpp>
#include <sstream>

namespace ov {
namespace frontend {
namespace onnx {

class PlaceInputEdge : public Place {
public:
    PlaceInputEdge(const InputEdge& edge, std::shared_ptr<ONNXModelEditor> editor);
    PlaceInputEdge(InputEdge&& edge, std::shared_ptr<ONNXModelEditor> editor);

    // internal usage
    InputEdge get_input_edge() const;
    void check_if_valid() const;

    // external usage
    std::vector<std::string> get_names() const override;
    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(const Place::Ptr& another) const override;
    bool is_equal_data(const Place::Ptr& another) const override;
    Place::Ptr get_source_tensor() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;
    Place::Ptr get_producing_operation() const override;
    Place::Ptr get_producing_port() const override;

private:
    InputEdge m_edge;
    const std::shared_ptr<ONNXModelEditor> m_editor;
    std::string m_initial_source_tensor_name;
};

class PlaceOutputEdge : public Place {
public:
    PlaceOutputEdge(const OutputEdge& edge, std::shared_ptr<ONNXModelEditor> editor);
    PlaceOutputEdge(OutputEdge&& edge, std::shared_ptr<ONNXModelEditor> editor);

    // internal usage
    OutputEdge get_output_edge() const;
    void check_if_valid() const;

    // external usage
    std::vector<std::string> get_names() const override;
    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(const Place::Ptr& another) const override;
    bool is_equal_data(const Place::Ptr& another) const override;
    Place::Ptr get_target_tensor() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Place::Ptr get_producing_operation() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;

private:
    OutputEdge m_edge;
    std::shared_ptr<ONNXModelEditor> m_editor;
    std::string m_initial_target_tensor_name;
};

class PlaceTensor : public Place {
public:
    PlaceTensor(const std::string& name, std::shared_ptr<ONNXModelEditor> editor);
    PlaceTensor(std::string&& name, std::shared_ptr<ONNXModelEditor> editor);

    // external usage
    std::vector<std::string> get_names() const override;
    Place::Ptr get_producing_port() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Place::Ptr get_producing_operation() const override;
    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(const Place::Ptr& another) const override;
    bool is_equal_data(const Place::Ptr& another) const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;

    void set_name(const std::string& new_name);
    void set_name_for_dimension(size_t shape_dim_index, const std::string& dim_name);

private:
    std::string m_name;
    std::shared_ptr<ONNXModelEditor> m_editor;
};

class PlaceOp : public Place {
public:
    PlaceOp(const EditorNode& node, std::shared_ptr<ONNXModelEditor> editor);
    PlaceOp(EditorNode&& node, std::shared_ptr<ONNXModelEditor> editor);
    std::vector<std::string> get_names() const override;

    // internal usage
    const EditorNode& get_editor_node() const;
    void set_name(const std::string& new_name);
    void check_if_valid() const;

    // external usage
    Place::Ptr get_output_port() const override;
    Place::Ptr get_output_port(int output_port_index) const override;
    Place::Ptr get_output_port(const std::string& output_port_name) const override;

    Place::Ptr get_input_port() const override;
    Place::Ptr get_input_port(int input_port_index) const override;
    Place::Ptr get_input_port(const std::string& input_name) const override;

    std::vector<Place::Ptr> get_consuming_ports() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;
    std::vector<Place::Ptr> get_consuming_operations(int output_port_index) const override;
    std::vector<Place::Ptr> get_consuming_operations(const std::string& output_port_name) const override;

    Place::Ptr get_producing_operation() const override;
    Place::Ptr get_producing_operation(int input_port_index) const override;
    Place::Ptr get_producing_operation(const std::string& input_port_name) const override;

    Place::Place::Ptr get_target_tensor() const override;
    Place::Ptr get_target_tensor(int output_port_index) const override;
    Place::Ptr get_target_tensor(const std::string& output_name) const override;

    Place::Place::Ptr get_source_tensor() const override;
    Place::Ptr get_source_tensor(int input_port_index) const override;
    Place::Ptr get_source_tensor(const std::string& input_name) const override;

    bool is_equal(const Place::Ptr& another) const override;
    bool is_input() const override;
    bool is_output() const override;

private:
    EditorNode m_node;
    std::shared_ptr<ONNXModelEditor> m_editor;
    std::string m_initial_first_output;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
