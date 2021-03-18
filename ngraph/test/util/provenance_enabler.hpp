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

#include "ngraph/provenance.hpp"

namespace ngraph
{
    namespace test
    {
        /// \brief Enable provenance for the duration of a unit test.
        ///
        /// During creation this object activates provenance support, when it's destroyed
        /// it returns the provenance support to previous state.
        class ProvenanceEnabler
        {
        public:
            ProvenanceEnabler()
            {
                saved_enable_state = get_provenance_enabled();
                set_provenance_enabled(true);
            }
            ~ProvenanceEnabler() { set_provenance_enabled(saved_enable_state); }

        private:
            bool saved_enable_state;
        };
    }
}
