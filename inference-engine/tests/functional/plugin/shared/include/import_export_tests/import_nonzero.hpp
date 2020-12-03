// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "import_export_base.hpp"

namespace LayerTestsDefinitions {

class ImportNonZero : public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
