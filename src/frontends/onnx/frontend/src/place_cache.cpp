#include "place_cache.hpp"

#include "place.hpp"

namespace ov {
namespace frontend {
namespace onnx {

PlaceCache::PlaceCache(std::shared_ptr<onnx_editor::ONNXModelEditor> editor) : m_editor{std::move(editor)} {}

Place::Ptr PlaceCache::get_tensor_place(std::string tensor_name) {
    auto tensor_place_it =
        std::find_if(std::begin(m_places), std::end(m_places), [&tensor_name](const Place::Ptr& place) {
            if (auto place_tensor = std::dynamic_pointer_cast<PlaceTensor>(place)) {
                return place_tensor->get_names().at(0) == tensor_name;
            }
            return false;
        });
    if (tensor_place_it == std::end(m_places)) {
        auto new_tensor_place =
            std::shared_ptr<PlaceTensor>(new PlaceTensor(tensor_name, m_editor, shared_from_this()));
        m_places.emplace_back(new_tensor_place);
        return new_tensor_place;
    } else {
        return *tensor_place_it;
    }
}

Place::Ptr PlaceCache::get_input_edge_place(onnx_editor::InputEdge edge) {
    auto in_place_place_it = std::find_if(std::begin(m_places), std::end(m_places), [&edge](const Place::Ptr& place) {
        if (auto in_place_tensor = std::dynamic_pointer_cast<PlaceInputEdge>(place)) {
            auto in_edge = in_place_tensor->get_input_edge();
            return (in_edge.m_node_idx == edge.m_node_idx && in_edge.m_port_idx == edge.m_port_idx);
        }
        return false;
    });
    if (in_place_place_it == std::end(m_places)) {
        auto new_in_place_place =
            std::shared_ptr<PlaceInputEdge>(new PlaceInputEdge(edge, m_editor, shared_from_this()));
        m_places.emplace_back(new_in_place_place);
        return new_in_place_place;
    } else {
        return *in_place_place_it;
    }
}

Place::Ptr PlaceCache::get_output_edge_place(onnx_editor::OutputEdge edge) {
    auto out_place_place_it = std::find_if(std::begin(m_places), std::end(m_places), [&edge](const Place::Ptr& place) {
        if (auto out_place_tensor = std::dynamic_pointer_cast<PlaceOutputEdge>(place)) {
            auto out_edge = out_place_tensor->get_output_edge();
            return (out_edge.m_node_idx == edge.m_node_idx && out_edge.m_port_idx == edge.m_port_idx);
        }
        return false;
    });
    if (out_place_place_it == std::end(m_places)) {
        auto new_out_place = std::shared_ptr<PlaceOutputEdge>(new PlaceOutputEdge(edge, m_editor, shared_from_this()));
        m_places.emplace_back(new_out_place);
        return new_out_place;
    } else {
        return *out_place_place_it;
    }
}

Place::Ptr PlaceCache::get_op_place(onnx_editor::EditorNode node) {
    auto op_place_place_it =
        std::find_if(std::begin(m_places), std::end(m_places), [&node, this](const Place::Ptr& place) {
            if (auto op_place_tensor = std::dynamic_pointer_cast<PlaceOp>(place)) {
                return m_editor->get_node_index(op_place_tensor->get_editor_node()) == m_editor->get_node_index(node);
            }
            return false;
        });
    if (op_place_place_it == std::end(m_places)) {
        auto new_op_place = std::shared_ptr<PlaceOp>(new PlaceOp(node, m_editor, shared_from_this()));
        m_places.emplace_back(new_op_place);
        return new_op_place;
    } else {
        return *op_place_place_it;
    }
}

std::vector<Place::Ptr> PlaceCache::get_cached_places() const {
    return m_places;
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov