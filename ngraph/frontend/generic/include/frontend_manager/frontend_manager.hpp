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

        };

        class NGRAPH_API InputModel
        {
        public:

            typedef std::shared_ptr<InputModel> Ptr;

            virtual std::vector<Place::Ptr> getInputs () const = 0;
            virtual std::vector<Place::Ptr> getOutputs () const = 0;
        };

        class NGRAPH_API FrontEnd
        {
        public:
            typedef std::shared_ptr<FrontEnd> Ptr;

            virtual InputModel::Ptr load (const std::string& path) const = 0;
            virtual std::shared_ptr<ngraph::Function>  convert (InputModel::Ptr model) const = 0;
        };

        class NGRAPH_API FrontEndManager
        {
        public:
            FrontEndManager () {}
            FrontEnd::Ptr loadByFramework (const std::string& framework);
            FrontEnd::Ptr loadByModel (const std::string& path);
            std::vector<std::string> availableFrontEnds () const;
        };
    } // namespace frontend

} // namespace ngraph
