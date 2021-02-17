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

#include <ngraph/except.hpp>
#include "onnx_import/onnx.hpp"
#include "onnx_import/editor/editor.hpp"
#include "frontend_manager/frontend_manager.hpp"

namespace ngraph
{
    namespace frontend
    {
        class InputModelONNX : public InputModel
        {
        public:
            onnx_import::ONNXModelEditor editor;

            InputModelONNX (const std::string& model_path) : editor(model_path) {}

            virtual std::vector<Place::Ptr> getInputs () const {
                throw ngraph::ngraph_error("getInputs is not supported for InputModelONNX");
            }

            virtual std::vector<Place::Ptr> getOutputs () const {
                throw ngraph::ngraph_error("getOutputs is not supported for InputModelONNX");
            }
        };

        class FrontEndONNX : public FrontEnd
        {
        public:

            FrontEndONNX ()
            {
            }

            virtual InputModel::Ptr load (const std::string& path) const {
                return std::make_shared<InputModelONNX>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const {
                return import_onnx_model(std::dynamic_pointer_cast<InputModelONNX>(model)->editor);
            }
        };


        FrontEnd::Ptr FrontEndManager::loadByFramework (const std::string& framework)
        {
            NGRAPH_CHECK(framework == "onnx");
            return std::make_shared<FrontEndONNX>();
        }

        FrontEnd::Ptr FrontEndManager::loadByModel (const std::string& path)
        {
            return loadByFramework("onnx");
        }

        std::vector<std::string> FrontEndManager::availableFrontEnds () const
        {
            return std::vector<std::string>(1, "onnx");
        }
    } // namespace frontend

} // namespace ngraph
