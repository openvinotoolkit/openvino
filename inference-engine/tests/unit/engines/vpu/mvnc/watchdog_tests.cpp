// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests_common.hpp>
#include <watchdog/watchdog.h>
#include <watchdog/watchdogPrivate.hpp>
#include <mvnc/include/ncPrivateTypes.h>
#include <thread>

using namespace ::testing;
using namespace InferenceEngine;

class MockWatchdogDevice : public Watchdog::IDevice {
 public:
    using time_point  = Watchdog::IDevice::time_point;
    MOCK_QUALIFIED_METHOD1(setInterval, noexcept, void(const std::chrono::milliseconds));
    MOCK_QUALIFIED_METHOD1(keepAlive, noexcept, void(const time_point &));
    MOCK_QUALIFIED_METHOD1(dueIn, const noexcept, std::chrono::milliseconds (const time_point &current_time));
    MOCK_QUALIFIED_METHOD0(isTimeout, const noexcept, bool ());
    MOCK_QUALIFIED_METHOD0(getHandle, const noexcept, void* ());
};

struct wd_context_opaque_private {
    void * magic = reinterpret_cast<void *> (0xdeadbeaf);
    Watchdog::IDevice * actual = nullptr;
    bool   destroyed = false;
};


class MVNCWatchdogTests: public TestsCommon {
 protected:
    devicePrivate_t d;
    wd_context ctx, ctx1;
    StrictMock<MockWatchdogDevice> mockWatchee, mockWatchee1;
    wd_context_opaque_private opaque, opaque1;

    void SetUp() override {
        opaque.actual = &mockWatchee;
        ctx.opaque = &opaque;

        opaque1.actual = &mockWatchee1;
        ctx1.opaque = &opaque1;

        pthread_mutex_init(&d.dev_stream_m, nullptr);
    }
    void TearDown() override {
        pthread_mutex_destroy(&d.dev_stream_m);
    }
};
using ms = std::chrono::milliseconds;

TEST_F(MVNCWatchdogTests, canRegisterExternalWatchee) {

    int handle = 1;
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&handle));
    // do not expect that  any ping happened before we remove the thread
    // this can be changed for example registering succeed only if first ping succeed
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(AtLeast(0));
    EXPECT_CALL(mockWatchee, setInterval(ms(1))).Times(1);
    EXPECT_CALL(mockWatchee, isTimeout()).WillRepeatedly(Return(false));
    EXPECT_CALL(mockWatchee, dueIn(_)).WillRepeatedly(Return(ms(20000)));

    d.wd_interval = 1;

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx, &d));
    // allowing thread spin
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
}

// TODO: implement logic
TEST_F(MVNCWatchdogTests, DISABLED_removeDeviceIfXLINKSessionNotIninitialized) {

    d.wd_interval = 10;
    ASSERT_EQ(WD_ERRNO, watchdog_init_context(&ctx));
    ASSERT_NE(WD_ERRNO, watchdog_register_device(&ctx, &d));

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}


TEST_F(MVNCWatchdogTests, canNotBeRegisteredTwice) {

    d.wd_interval = 10;

    ASSERT_EQ(WD_ERRNO, watchdog_init_context(&ctx));
    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx, &d));
    ASSERT_NE(WD_ERRNO, watchdog_register_device(&ctx, &d));
    // allowing thread spin
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
}

TEST_F(MVNCWatchdogTests, canUnRegisterNotInitialized) {

    ASSERT_EQ(WD_ERRNO, watchdog_init_context(&ctx));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
}

TEST_F(MVNCWatchdogTests, canUnRegisterIfInterval0) {

    d.wd_interval = 0;

    ASSERT_EQ(WD_ERRNO, watchdog_init_context(&ctx));
    ASSERT_NE(WD_ERRNO, watchdog_register_device(&ctx, &d));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
}

TEST_F(MVNCWatchdogTests, failUnRegisterTwice) {

    d.wd_interval = 10;

    ASSERT_EQ(WD_ERRNO, watchdog_init_context(&ctx));
    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx, &d));
    // allowing thread spin
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
    ASSERT_NE(WD_ERRNO, watchdog_unregister_device(&ctx));
}

TEST_F(MVNCWatchdogTests, canRemoveOneDeviceFromQueueInCaseOfTimeout) {
    int handle = 1;
    int x = 0;
    int y = 0;
    int z = 0;
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&handle));
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(AtLeast(1));
    EXPECT_CALL(mockWatchee, setInterval(ms(10))).Times(1);
    EXPECT_CALL(mockWatchee, isTimeout()).WillRepeatedly(Invoke([&z, &y]() {
        // will sleep at least 100 ms and avoid second keep alive call
        y = 100;
        if (!z) {
            // sleep in watchdog thread, and allowing register second device before deleting first one
            z = 1;
            return false;
        }
        return true;
    }));
    EXPECT_CALL(mockWatchee, dueIn(_)).WillRepeatedly(Invoke([&y](const MockWatchdogDevice::time_point &current_time){
        return std::chrono::milliseconds(y);
    }));

    EXPECT_CALL(mockWatchee1, getHandle()).WillRepeatedly(Return(&handle));
    EXPECT_CALL(mockWatchee1, keepAlive(_)).Times(AtLeast(2));
    EXPECT_CALL(mockWatchee1, setInterval(ms(10))).Times(1);
    EXPECT_CALL(mockWatchee1, isTimeout()).WillRepeatedly(Invoke([&x]() {
        // allow every second time to wait
        x = x == 0 ? 100 : 0;
        return false;
    }));
    EXPECT_CALL(mockWatchee1, dueIn(_)).WillRepeatedly(Invoke([&x](const MockWatchdogDevice::time_point &current_time){
        return std::chrono::milliseconds(x);
    }));


    d.wd_interval = 10;

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx, &d));
    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx1, &d));

    std::this_thread::sleep_for(ms(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx1));
}

TEST_F(MVNCWatchdogTests, canNotStartWatchdogIfIntervalInvalid) {

    opaque.actual = &mockWatchee;

    int handle = 1;

    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&handle));

    d.wd_interval = 0;
    ASSERT_NE(WD_ERRNO, watchdog_register_device(&ctx, &d));

    d.wd_interval = -1;
    ASSERT_NE(WD_ERRNO, watchdog_register_device(&ctx, &d));

    // if fo some reason thread started we will get unxpected updatePongInterval calls
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

TEST_F(MVNCWatchdogTests, canGetPingsOnRegularBasis) {

    int handle = 1;
    int x = 0;
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&handle));
    // since interval is small keepAlive can happen several times once
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(AtLeast(2));
    EXPECT_CALL(mockWatchee, setInterval(ms(10))).Times(1);
    EXPECT_CALL(mockWatchee, isTimeout()).WillRepeatedly(Return(false));
    EXPECT_CALL(mockWatchee, dueIn(_)).WillRepeatedly(Invoke([&x](const MockWatchdogDevice::time_point &current_time){
        x = x == 0 ? 100 : 0;
        return std::chrono::milliseconds(x);
    }));


    d.wd_interval = 10;

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx, &d));

    std::this_thread::sleep_for(ms(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
}

TEST_F(MVNCWatchdogTests, canWakeUpWatchdogWhenAddAndRemoveDevice) {

    int handle = 1, handle1 = 2;

    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&handle));
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(1);
    EXPECT_CALL(mockWatchee, setInterval(ms(10))).Times(1);
    EXPECT_CALL(mockWatchee, isTimeout()).WillRepeatedly(Return(false));
    // without wake this will sleep for ever
    EXPECT_CALL(mockWatchee, dueIn(_)).WillRepeatedly(Return(ms(20000)));

    EXPECT_CALL(mockWatchee1, getHandle()).WillRepeatedly(Return(&handle1));
    EXPECT_CALL(mockWatchee1, keepAlive(_)).Times(1);
    EXPECT_CALL(mockWatchee1, setInterval(ms(10))).Times(1);
    EXPECT_CALL(mockWatchee1, isTimeout()).WillRepeatedly(Return(false));
    EXPECT_CALL(mockWatchee1, dueIn(_)).WillRepeatedly(Return(ms(20000)));


    d.wd_interval = 10;

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx, &d));

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx1, &d));

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx1));
}

TEST_F(MVNCWatchdogTests, stressWatchDog) {

    const int num_watchdog_device = 10;

    watchdog_init_context(nullptr);

    StrictMock<MockWatchdogDevice> mockWatchee[num_watchdog_device];
    int handle[num_watchdog_device];
    wd_context ctx[num_watchdog_device];
    wd_context_opaque_private opaque[num_watchdog_device];

    for (int i = 0; i != num_watchdog_device; i++) {
        handle[i] = i;

        EXPECT_CALL(mockWatchee[i], getHandle()).WillRepeatedly(Return(handle + i));
        // since interval is big keepAlive happens only once
        EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(1);

        EXPECT_CALL(mockWatchee[i], setInterval(ms(10))).Times(1);
        EXPECT_CALL(mockWatchee[i], isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mockWatchee[i], dueIn(_)).WillRepeatedly(Return(ms(20000)));
    }

    d.wd_interval = 10;

    for (int k = 0; k != num_watchdog_device; k++) {
        opaque[k].actual = &mockWatchee[k];
        ctx[k].opaque = &opaque[k];
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx[k], &d));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(MVNCWatchdogTests, stressWatchDog1) {

    const int num_watchdog_device = 10;
    const int num_watchdog_device_half = num_watchdog_device / 2;

    watchdog_init_context(nullptr);

    StrictMock<MockWatchdogDevice> mockWatchee[num_watchdog_device];
    int handle[num_watchdog_device];
    wd_context ctx[num_watchdog_device];
    wd_context_opaque_private opaque[num_watchdog_device];

    for (int i = 0; i != num_watchdog_device; i++) {
        handle[i] = i;

        EXPECT_CALL(mockWatchee[i], getHandle()).WillRepeatedly(Return(handle + i));
        // since interval is big keepAlive happens only once
        EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(1);

        EXPECT_CALL(mockWatchee[i], setInterval(ms(10))).Times(1);
        EXPECT_CALL(mockWatchee[i], isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mockWatchee[i], dueIn(_)).WillRepeatedly(Return(ms(20000)));
    }

    d.wd_interval = 10;
    for (int k = 0; k != num_watchdog_device; k++) {
        opaque[k].actual = &mockWatchee[k];
        ctx[k].opaque = &opaque[k];
    }

    for (int k = 0; k != num_watchdog_device_half; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx[k], &d));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device_half; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx[k + num_watchdog_device_half], &d));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx[k]));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device_half; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx[k + num_watchdog_device_half]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(MVNCWatchdogTests, stressWatchDog2) {

    const int num_watchdog_device = 30;
    const int num_watchdog_device_half1 = num_watchdog_device / 3;
    const int num_watchdog_device_half2 = 2 * num_watchdog_device / 3;

    watchdog_init_context(nullptr);

    StrictMock<MockWatchdogDevice> mockWatchee[num_watchdog_device];
    int handle[num_watchdog_device];
    wd_context ctx[num_watchdog_device];
    wd_context_opaque_private opaque[num_watchdog_device];

    for (int i = 0; i != num_watchdog_device; i++) {
        handle[i] = i;

        EXPECT_CALL(mockWatchee[i], getHandle()).WillRepeatedly(Return(handle + i));

        // since interval is big keepAlive happens only once
        if (i >= num_watchdog_device_half2) {
            EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(AtLeast(0));
        } else {
            EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(1);
        }

        EXPECT_CALL(mockWatchee[i], setInterval(ms(10))).Times(1);
        EXPECT_CALL(mockWatchee[i], isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mockWatchee[i], dueIn(_)).WillRepeatedly(Return(ms(20000)));
    }

    d.wd_interval = 10;
    for (int k = 0; k != num_watchdog_device; k++) {
        opaque[k].actual = &mockWatchee[k];
        ctx[k].opaque = &opaque[k];
    }

    for (int k = 0; k != num_watchdog_device_half1; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx[k], &d));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device_half1; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = num_watchdog_device_half1; k != num_watchdog_device_half2; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx[k], &d));
        //this might lead to UB, for example thread might restart but after that device get removed, so giving more time
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx[k]));
    }

    for (int k = num_watchdog_device_half2; k != num_watchdog_device; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(&ctx[k], &d));
        //this might lead to UB, for example thread might restart but after that device get removed, so giving more time
        //so our expectations for number of calls are not set for last third
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(&ctx[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
}
