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

#include <memory>
#include <string>
#include "ngraph/function.hpp"
#include "ngraph/visibility.hpp"

namespace ngraph
{
    namespace frontend
    {
        class NGRAPH_API Place
        {
        public:

            typedef std::shared_ptr<Place> Ptr;

            // All associated names that uniquely identify this place in the graph
            // from the FW perspective
            virtual std::vector<std::string> getNames () const;
            // -1 means port 0 is selected if it is exists and exception otherwise
            virtual Ptr getConsumingOperation (int outputPortIndex = -1) const;
            virtual Ptr getConsumingTensor (int outputPortIndex = -1) const;
            virtual Ptr getProducingOperation (int inputPortIndex = -1) const;
            virtual Ptr getProducingPort () const;
            virtual Ptr getInputPort (int inputPortIndex) const;
            virtual Ptr getOutputPort (int outputPortIndex) const;
            virtual std::vector<Ptr> getConsumingPorts () const;
        };

        class NGRAPH_API InputModel
        {
        public:

            typedef std::shared_ptr<InputModel> Ptr;

            virtual std::vector<Place::Ptr> getInputs () const;
            virtual std::vector<Place::Ptr> getOutputs () const;

            virtual Place::Ptr getPlaceByTensorName (const std::string& tensorName);
            virtual Place::Ptr getPlaceByOperationName (const std::string& operationName);
            virtual Place::Ptr getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex);
            virtual Place::Ptr getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex);

            // Naming and annotation
            virtual void setNameForTensor (Place::Ptr tensor, const std::string& newName);
            virtual void addNameForTensor (Place::Ptr tensor, const std::string& newName);
            virtual void setNameForOperation (Place::Ptr operation, const std::string& newName);
            virtual void freeNameForTensor (const std::string& name);
            virtual void freeNameForOperation (const std::string& name);
            virtual void setNameForDimension (Place::Ptr place, const std::string& dimName);

            // Topology Editing
            virtual void cutAndAddNewInput (Place::Ptr place, const std::string& newNameOptional = "");
            virtual void cutAndAddNewOutput (Place::Ptr place, const std::string& newNameOptional = "");
            virtual void addOutput (Place::Ptr place);
            virtual void removeOutput (Place::Ptr place);
            virtual void removeInput (Place::Ptr place);    // is it really needed? that means that input is not available and all dataflow below should be removed
            virtual void overrideAllOutputs (const std::vector<Place::Ptr>& outputs);
            virtual void overrideAllInputs (const std::vector<Place::Ptr>& inputs);
            virtual void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);

            // Setting tensor properties
            virtual void setDefaultShape (Place::Ptr place, const ngraph::Shape&);
            virtual void setPartialShape (Place::Ptr place, const ngraph::PartialShape&);
            virtual void setElementType (Place::Ptr place, const ngraph::element::Type&);
            virtual void setTensorValue (Place::Ptr place, const void* value);
            virtual void setTensorPartialValue (Place::Ptr place, const void* minValue, const void* maxValue);

            // Standard specializations: "N", "C", "H", "W", "D", "L"
            // Issues: name collisions; too bulky to select simple layouts "NCHW"
            virtual void setSourceTensorDimSpecialization (Place::Ptr place, unsigned int dimIndex, const std::string& specialization);
            virtual void setTargetTensorDimSpecialization (Place::Ptr place, unsigned int dimIndex, const std::string& specialization);

            // Traversing
            // TODO

            // Support querying
            // TODO
        };

        class NGRAPH_API FrontEnd
        {
        public:
            typedef std::shared_ptr<FrontEnd> Ptr;

            // TODO: rename to loadFromPath
            virtual InputModel::Ptr load (const std::string& path) const = 0;
            virtual InputModel::Ptr loadFromPaths (const std::vector<std::string>& paths) const;
            virtual InputModel::Ptr loadFromMemory (const void* model);
            virtual InputModel::Ptr loadFromMemoryFragments (const std::vector<const void*> modelParts);
            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const = 0;

            // Convert only those parts of the model that
            virtual std::shared_ptr<ngraph::Function> convertPartially (InputModel::Ptr model) const;
            virtual void normalize (std::shared_ptr<ngraph::Function> function) const;
        };

        enum FrontEndCapabilities {
            FEC_DEFAULT   =  0,    // Just reading and conversion, w/o any modifications; intended to be used in Reader
            FEC_CUT       =  1,
            FEC_NAMES     =  2,
            FEC_REPLACE   =  4,
            FEC_TRAVERSE  =  8,
            FEC_WILDCARDS = 16,
        };

        class NGRAPH_API FrontEndManager
        {
        public:
            FrontEndManager () {}
            FrontEnd::Ptr loadByFramework (const std::string& framework, FrontEndCapabilities fec = FEC_DEFAULT);
            FrontEnd::Ptr loadByModel (const std::string& path, FrontEndCapabilities fec = FEC_DEFAULT);
            std::vector<std::string> availableFrontEnds () const;
        };
    } // namespace frontend

} // namespace ngraph
