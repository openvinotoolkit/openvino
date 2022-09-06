// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "meta_data.hpp"

namespace ov {

class DefaultMetaData : public ov::Meta {
public:
    operator ov::AnyMap&() override {
        return m_map;
    }
    operator const ov::AnyMap&() const override {
        return m_map;
    }

private:
    ov::AnyMap m_map;
};
}  // namespace ov
