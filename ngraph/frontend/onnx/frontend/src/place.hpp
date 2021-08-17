// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <editor.hpp>
#include <frontend_manager/place.hpp>
#include <memory>

namespace ngraph {
namespace frontend {
class PlaceInputEdgeONNX : public Place {
public:
    PlaceInputEdgeONNX(const onnx_editor::InputEdge& edge, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    onnx_editor::InputEdge get_input_edge() const;

    bool is_input() const override;

    bool is_output() const override;

    bool is_equal(Place::Ptr another) const override;

    bool is_equal_data(Place::Ptr another) const override;

    Place::Ptr get_source_tensor() const override;

private:
    onnx_editor::InputEdge m_edge;
    const std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};

class PlaceOutputEdgeONNX : public Place {
public:
    PlaceOutputEdgeONNX(const onnx_editor::OutputEdge& edge, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    onnx_editor::OutputEdge get_output_edge() const;

    bool is_input() const override;

    bool is_output() const override;

    bool is_equal(Place::Ptr another) const override;

    bool is_equal_data(Place::Ptr another) const override;

    Place::Ptr get_target_tensor() const override;

private:
    onnx_editor::OutputEdge m_edge;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};

class PlaceTensorONNX : public Place {
public:
    PlaceTensorONNX(const std::string& name, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    std::vector<std::string> get_names() const override;

    Place::Ptr get_producing_port() const override;

    std::vector<Place::Ptr> get_consuming_ports() const override;

    Ptr get_input_port(int input_port_index) const override;

    bool is_input() const override;

    bool is_output() const override;

    bool is_equal(Place::Ptr another) const override;

    bool is_equal_data(Place::Ptr another) const override;

private:
    std::string m_name;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};
}  // namespace frontend

}  // namespace ngraph
