// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace SubgraphsDumper {
struct Model {
    std::string xml;
    std::string bin;
    size_t size;

    Model(std::string model) {
        xml = model;
        bin = CommonTestUtils::replaceExt(model, "bin");
        if (CommonTestUtils::fileExists(bin)) {
            size = CommonTestUtils::fileSize(bin);
        } else {
            size = 0;
        }
    }

    bool operator<(const Model &m) const {
        return size < m.size;
    }

    bool operator>(const Model &m) const {
        return size > m.size;
    }
};
}  // namespace SubgraphsDumper