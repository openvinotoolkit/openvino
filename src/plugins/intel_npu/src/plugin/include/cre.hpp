// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "isection.hpp"

namespace intel_npu {

using CREToken = uint16_t;

class CRESection final : public ISection {
public:
    static constexpr SectionID id = 100;

    CRESection();

    void write(std::ostream& stream, BlobWriter* writer) override;

    // void read(BlobReader* reader) override;

    void append_to_expression(const CREToken requirement_token);

private:
    std::vector<CREToken> expression;
};

}  // namespace intel_npu
