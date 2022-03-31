// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <editor.hpp>
#include <memory>
#include <openvino/frontend/place.hpp>
#include <sstream>

#include "place_cache.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class PlaceOnnx : public Place {
public:
    void invalidate();
    void check_if_valid() const;

private:
    bool m_invalidated = false;
};

class PlaceInputEdge : public PlaceOnnx {
public:
    // internal usage
    onnx_editor::InputEdge get_input_edge() const;

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
    friend class PlaceCache;
    PlaceInputEdge(const onnx_editor::InputEdge& edge,
                   std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
                   std::shared_ptr<PlaceCache> place_cache);
    PlaceInputEdge(onnx_editor::InputEdge&& edge,
                   std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
                   std::shared_ptr<PlaceCache> place_cache);

private:
    onnx_editor::InputEdge m_edge;
    const std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
    std::shared_ptr<PlaceCache> m_place_cache;
};

class PlaceOutputEdge : public PlaceOnnx {
public:
    // internal usage
    onnx_editor::OutputEdge get_output_edge() const;

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
    friend class PlaceCache;
    PlaceOutputEdge(const onnx_editor::OutputEdge& edge,
                    std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
                    std::shared_ptr<PlaceCache> place_cache);
    PlaceOutputEdge(onnx_editor::OutputEdge&& edge,
                    std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
                    std::shared_ptr<PlaceCache> place_cache);

private:
    onnx_editor::OutputEdge m_edge;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
    std::shared_ptr<PlaceCache> m_place_cache;
};

class PlaceTensor : public PlaceOnnx {
public:
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
    friend class PlaceCache;
    PlaceTensor(const std::string& name,
                std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
                std::shared_ptr<PlaceCache> place_cache);
    PlaceTensor(std::string&& name,
                std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
                std::shared_ptr<PlaceCache> place_cache);

private:
    std::string m_name;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
    std::shared_ptr<PlaceCache> m_place_cache;
};

class PlaceOp : public PlaceOnnx {
public:
    std::vector<std::string> get_names() const override;

    // internal usage
    const onnx_editor::EditorNode& get_editor_node() const;
    void set_name(const std::string& new_name);

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
    friend class PlaceCache;
    PlaceOp(const onnx_editor::EditorNode& node,
            std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
            std::shared_ptr<PlaceCache> place_cache);
    PlaceOp(onnx_editor::EditorNode&& node,
            std::shared_ptr<onnx_editor::ONNXModelEditor> editor,
            std::shared_ptr<PlaceCache> place_cache);

private:
    onnx_editor::EditorNode m_node;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
    std::shared_ptr<PlaceCache> m_place_cache;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
