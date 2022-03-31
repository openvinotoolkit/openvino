// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <openvino/frontend/place.hpp>
#include <vector>

#include "editor.hpp"
#include "editor_types.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class PlaceCache : public std::enable_shared_from_this<PlaceCache> {
public:
    PlaceCache(std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    // TODO: Consider handling r-value string
    Place::Ptr get_tensor_place(std::string tensor_name);
    Place::Ptr get_input_edge_place(onnx_editor::InputEdge edge);
    Place::Ptr get_output_edge_place(onnx_editor::OutputEdge edge);
    Place::Ptr get_op_place(onnx_editor::EditorNode node);

    std::vector<Place::Ptr> get_cached_places() const;

private:
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
    std::vector<Place::Ptr> m_places;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov