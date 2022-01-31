// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/numeric.hpp>


#include <memory>
#include <cstdint>

namespace vpu {

class DataContent {
public:
    using Ptr = std::shared_ptr<DataContent>;
    using CPtr = std::shared_ptr<const DataContent>;

    virtual ~DataContent();

    template<typename T>
    const T* get() const {
        return static_cast<const T*>(getRaw());
    }

    virtual size_t byteSize() const = 0;

private:
    virtual const void* getRaw() const = 0;
};

} // namespace vpu
