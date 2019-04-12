//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "gna_plugin/gna_allocator.hpp"

#include <vector>
#include <thread>

#include <gtest/gtest.h>
#include "gna_plugin/gna_device.hpp"
//dummy definitions to work around issue with Linux userspace library
typedef unsigned long long time_tsc;
typedef struct
{
    time_tsc            start;      // time value on profiler start
    time_tsc            stop;       // time value on profiler stop
    time_tsc            passed;     // time passed between start and stop
} intel_gna_profiler_tsc;

void profilerTscStop(intel_gna_profiler_tsc* p) {
    if (NULL == p) return;
    p->passed = 0;
    p->stop = 0;
    p->start = 0;
}
void profilerTscStartAccumulate(intel_gna_profiler_tsc* p)
{
    if (NULL == p) return;
    p->stop = 0;
    p->start = 0;
}
void profilerTscStopAccumulate(intel_gna_profiler_tsc* p)
{
    if (NULL == p) return;
    p->stop = 0;
    p->passed += p->stop - p->start;
}

class GNAAllocatorTest : public ::testing::Test {

 protected:
    std::unique_ptr<GNADeviceHelper> gnadevice;
    void SetUp() override  {
       // gnadevice.reset(new GNADeviceHelper());
    }
};

TEST_F(GNAAllocatorTest, canAllocateStdMemory) {
    auto sp = make_polymorph<std::allocator<uint8_t>>();
    uint8_t *x = nullptr;
    ASSERT_NO_THROW(x = sp.allocate(100));
    ASSERT_NE(x, nullptr);
    ASSERT_NO_THROW(sp.deallocate(x, 100));
}

TEST_F(GNAAllocatorTest, canAllocateGNAMemory) {
    //GNA device can be opened one per process for now
    gnadevice.reset(new GNADeviceHelper());
    auto sp = make_polymorph<GNAAllocator>(*gnadevice.get());
    uint8_t *x = nullptr;
    ASSERT_NO_THROW(x = sp.allocate(100));
    ASSERT_NE(x, nullptr);
    ASSERT_NO_THROW(sp.deallocate(x, 100));
}

TEST_F(GNAAllocatorTest, DISABLED_canOpenDevice) {
    std::thread th([]()
    {
        GNADeviceHelper h1;
    });
    th.join();
    std::thread th2([]()
   {
       GNADeviceHelper h1;
   });
    th2.join();
}
