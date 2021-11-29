// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
