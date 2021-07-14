// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/place.hpp>
#include <onnx_editor/editor.hpp>

namespace ngraph
{
    namespace frontend
    {
        class FRONTEND_API PlaceInputEdgeONNX : public Place
        {
        public:
            PlaceInputEdgeONNX(const onnx_editor::InputEdge& edge,
                               const onnx_editor::ONNXModelEditor& editor)
                : m_edge{edge}
                , m_editor{editor}
            {
            }

            onnx_editor::InputEdge get_input_edge() const { return m_edge; }

            bool is_input() const override { return m_editor.is_input(m_edge); }

            bool is_output() const override { return false; }

        private:
            onnx_editor::InputEdge m_edge;
            const onnx_editor::ONNXModelEditor& m_editor;
        };

        class FRONTEND_API PlaceOutputEdgeONNX : public Place
        {
        public:
            PlaceOutputEdgeONNX(const onnx_editor::OutputEdge& edge,
                                const onnx_editor::ONNXModelEditor& editor)
                : m_edge{edge}
                , m_editor{editor}
            {
            }

            onnx_editor::OutputEdge get_output_edge() const { return m_edge; }

            bool is_input() const override { return false; }

            bool is_output() const override { return m_editor.is_output(m_edge); }

        private:
            onnx_editor::OutputEdge m_edge;
            const onnx_editor::ONNXModelEditor& m_editor;
        };

        class FRONTEND_API PlaceTensorONNX : public Place
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
                return std::make_shared<PlaceOutputEdgeONNX>(m_editor.find_output_edge(m_name),
                                                             m_editor);
            }

            Place::Ptr get_producing_operation(int input_port_index) const override
            {
                const auto edge =
                    m_editor.find_input_edge(onnx_editor::EditorOutput(m_name), input_port_index);
                return std::make_shared<PlaceInputEdgeONNX>(edge, m_editor);
            }

            Place::Ptr get_producing_operation() const override
            {
                // TODO: CHECK Number of ports
                return get_producing_operation(0);
            }

            std::vector<Place::Ptr> get_consuming_ports() const override
            {
                std::vector<Place::Ptr> ret;
                auto edges = m_editor.find_output_consumers(m_name);
                std::transform(edges.begin(),
                               edges.end(),
                               std::back_inserter(ret),
                               [this](const onnx_editor::InputEdge& edge) {
                                   return std::make_shared<PlaceInputEdgeONNX>(edge,
                                                                               this->m_editor);
                               });
                return ret;
            }

            Ptr get_input_port(int input_port_index) const override
            {
                return std::make_shared<PlaceInputEdgeONNX>(
                    m_editor.find_input_edge(onnx_editor::EditorNode(m_name),
                                             onnx_editor::EditorInput(input_port_index)),
                    m_editor);
            }

            bool is_input() const override
            {
                const auto inputs = m_editor.model_inputs();
                return std::find(std::begin(inputs), std::end(inputs), m_name) != std::end(inputs);
            }

            bool is_output() const override
            {
                const auto outputs = m_editor.model_outputs();
                return std::find(std::begin(outputs), std::end(outputs), m_name) !=
                       std::end(outputs);
            }

        private:
            std::string m_name;
            const onnx_editor::ONNXModelEditor& m_editor;
        };
    } // namespace frontend

} // namespace ngraph
