// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/place.hpp>

namespace ngraph
{
    namespace frontend
    {
        class PlaceInputEdgeONNX : public Place
        {
        public:
            PlaceInputEdgeONNX(const onnx_editor::InputEdge& edge)
                : m_edge(edge)
            {
            }

        private:
            onnx_editor::InputEdge m_edge;
        };

        class PlaceOutputEdgeONNX : public Place
        {
        public:
            PlaceOutputEdgeONNX(const onnx_editor::OutputEdge& edge)
                : m_edge(edge)
            {
            }

        private:
            onnx_editor::OutputEdge m_edge;
        };

        class PlaceTensorONNX : public Place
        {
        public:
            PlaceTensorONNX(const std::string& name, const onnx_editor::ONNXModelEditor& editor)
                : m_name(name)
                , m_editor(editor)
            {
            }

            std::vector<std::string> get_names() const override { return {m_name}; }

            Place::Ptr get_producing_port() const override
            {
                return std::make_shared<PlaceOutputEdgeONNX>(m_editor.find_output_edge(m_name));
            }

            std::vector<Place::Ptr> get_consuming_ports() const override
            {
                std::vector<Place::Ptr> ret;
                auto edges = m_editor.find_output_consumers(m_name);
                std::transform(edges.begin(),
                               edges.end(),
                               std::back_inserter(ret),
                               [](const onnx_editor::InputEdge& edge) {
                                   return std::make_shared<PlaceInputEdgeONNX>(edge);
                               });
                return ret;
            }

            Ptr get_input_port(int input_port_index) const override
            {
                return std::make_shared<PlaceInputEdgeONNX>(m_editor.find_input_edge(
                    onnx_editor::EditorNode(m_name), onnx_editor::EditorInput(input_port_index)));
            }

        private:
            std::string m_name;
            const onnx_editor::ONNXModelEditor& m_editor;
        };
    } // namespace frontend

} // namespace ngraph
