//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

// TODO: include it by just frontend_manager.hpp without path
#include <frontend_manager/frontend.hpp>

#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)

#define TF_API NGRAPH_HELPER_DLL_EXPORT

namespace tensorflow { class GraphDef; class NodeDef; namespace ngraph_bridge { class GraphIteratorProto; }  }

namespace ngraph
{
    namespace frontend
    {

#if 0
        class TFOperatorExtension
        {
        public:

            TFOperatorExtension (const std::string& optype, function<ngraph::OutputVector(const ngraph::frontend::tensorflow::NodeContext&)> converter);

        };
#endif

        class PlaceTensorflow : public Place
        {
        public:

            std::string name;
            enum Kind { PORT_INPUT, PORT_OUTPUT, TENSOR, OP } kind;
            size_t port;

            PlaceTensorflow (const std::string& _name, Kind _kind = OP, size_t _port = 0) : name(_name), kind(_kind), port(_port) {}

            virtual std::vector<std::string> get_names () const override { return {name}; }

            virtual bool is_equal(Ptr another) const override
            {
                auto another_tf = std::dynamic_pointer_cast<PlaceTensorflow>(another);
                return another_tf && name == another_tf->name && kind == another_tf->kind && port == another_tf->port;
            }
        };

        class TF_API InputModelTensorflow : public InputModel
        {
        public:

            std::shared_ptr<::tensorflow::ngraph_bridge::GraphIteratorProto> graph_impl;

            std::shared_ptr<::tensorflow::GraphDef> graph_def;
            std::string path;
            std::vector<ngraph::PartialShape> input_shapes;

            // TODO: map from PlaceTensorflow, not from name string
            std::map<std::string, ngraph::PartialShape> partialShapes;

            InputModelTensorflow (const std::string& _path);
            InputModelTensorflow (std::shared_ptr<::tensorflow::GraphDef> _graph_def, std::vector<ngraph::PartialShape> _input_shapes = {});
            InputModelTensorflow (const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _nodes_def, std::vector<ngraph::PartialShape> _input_shapes = {});

            std::vector<Place::Ptr> get_inputs () const override;

            virtual void set_partial_shape (Place::Ptr place, const ngraph::PartialShape& pshape) override;
            virtual ngraph::PartialShape get_partial_shape (Place::Ptr place) const override;
        };

        class TF_API FrontEndTensorflow : public FrontEnd
        {
        public:

            //using Converter = std::function<ngraph::OutputVector(const ngraph::frontend::tensorflow::NodeContext&)>;

            //void register_converter (const std::string& op_type, const Converter&);

            FrontEndTensorflow ()
            {
            }

            virtual InputModel::Ptr load_from_file (const std::string& path) const
            {
                return std::make_shared<InputModelTensorflow>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const override;

protected:
            InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override {
                if (variants.size() == 1) {
                    // The case when folder with __model__ and weight files is provided or .pdmodel file
                    if (is_type<VariantWrapper<std::string>>(variants[0])) {
                        std::string m_path = as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
                        return std::make_shared<InputModelTensorflow>(m_path);
                    }
                }
                return nullptr;
            }
        };

    } // namespace frontend

} // namespace ngraph
