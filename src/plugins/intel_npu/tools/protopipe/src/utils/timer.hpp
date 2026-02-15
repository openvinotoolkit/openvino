//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <functional>
#include <memory>

struct IWaitable {
    using Ptr = std::shared_ptr<IWaitable>;
    virtual void wait(std::chrono::microseconds time) = 0;
    virtual ~IWaitable() = default;
};

struct SleepTimer : public IWaitable {
    using Ptr = std::shared_ptr<SleepTimer>;
    static Ptr create(bool disable_high_resolution_timer = false);
};

struct BusyTimer : public IWaitable {
    void wait(std::chrono::microseconds time) override;
};
