// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests_common.hpp>
#include <watchdog/watchdog.h>
#include <watchdog/watchdogPrivate.hpp>
#include <thread>

using namespace ::testing;
using namespace InferenceEngine;

using ms = std::chrono::milliseconds;

class MockWatchdogDevice : public Watchdog::IDevice {
 public:
    using time_point  = Watchdog::IDevice::time_point;
    MOCK_METHOD(void, keepAlive, (const time_point &), (noexcept));
    MOCK_METHOD(std::chrono::milliseconds, dueIn, (const time_point &current_time), (const, noexcept));
    MOCK_METHOD(bool, isTimeout, (), (const, noexcept));
    MOCK_METHOD(void *, getHandle, (), (const, noexcept));
};

class MVNCWatchdogTests: public TestsCommon {
 protected:
    WatchdogHndl_t* m_watchdogHndl = nullptr;
    WdDeviceHndl_t deviceHndl, deviceHndl1;
    StrictMock<MockWatchdogDevice> mockWatchee, mockWatchee1;

    void SetUp() override {
        deviceHndl.m_device = &mockWatchee;
        deviceHndl1.m_device = &mockWatchee1;

        ASSERT_EQ(WD_ERRNO, watchdog_create(&m_watchdogHndl));
    }

    void TearDown() override {
        watchdog_destroy(m_watchdogHndl);
    }

    void setExpectations(StrictMock<MockWatchdogDevice>& mock){
        EXPECT_CALL(mock, keepAlive(_)).Times(AtLeast(0));
        EXPECT_CALL(mock, dueIn(_)).WillRepeatedly(Return(ms(20000)));
        EXPECT_CALL(mock, isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mock, getHandle()).WillRepeatedly(Return(&mock));
    }
};

TEST_F(MVNCWatchdogTests, canRegisterExternalWatchee) {
    setExpectations(mockWatchee);

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));
    // allowing thread spin
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
}

TEST_F(MVNCWatchdogTests, canNotBeRegisteredTwice) {
    setExpectations(mockWatchee);

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));
    ASSERT_NE(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));

    // allowing thread spin
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
}

TEST_F(MVNCWatchdogTests, canNotUnRegisterNotInitialized) {
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&mockWatchee));

    ASSERT_NE(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
}

TEST_F(MVNCWatchdogTests, failUnRegisterTwice) {
    setExpectations(mockWatchee);

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));

    // allowing thread spin
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
    ASSERT_NE(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
}

TEST_F(MVNCWatchdogTests, canRemoveOneDeviceFromQueueInCaseOfTimeout) {
    int x = 0;
    int y = 0;
    int z = 0;
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&mockWatchee));
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(AtLeast(1));
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

    EXPECT_CALL(mockWatchee1, getHandle()).WillRepeatedly(Return(&mockWatchee1));
    EXPECT_CALL(mockWatchee1, keepAlive(_)).Times(AtLeast(2));
    EXPECT_CALL(mockWatchee1, isTimeout()).WillRepeatedly(Invoke([&x]() {
        // allow every second time to wait
        x = x == 0 ? 100 : 0;
        return false;
    }));
    EXPECT_CALL(mockWatchee1, dueIn(_)).WillRepeatedly(Invoke([&x](const MockWatchdogDevice::time_point &current_time){
        return std::chrono::milliseconds(x);
    }));

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));
    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl1));

    std::this_thread::sleep_for(ms(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl1));
}

TEST_F(MVNCWatchdogTests, canGetPingsOnRegularBasis) {
    int x = 0;
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&mockWatchee));
    // since interval is small keepAlive can happen several times once
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(AtLeast(2));
    EXPECT_CALL(mockWatchee, isTimeout()).WillRepeatedly(Return(false));
    EXPECT_CALL(mockWatchee, dueIn(_)).WillRepeatedly(Invoke([&x](const MockWatchdogDevice::time_point &current_time){
        x = x == 0 ? 100 : 0;
        return std::chrono::milliseconds(x);
    }));

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));

    std::this_thread::sleep_for(ms(1000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
}

TEST_F(MVNCWatchdogTests, canWakeUpWatchdogWhenAddAndRemoveDevice) {
    EXPECT_CALL(mockWatchee, getHandle()).WillRepeatedly(Return(&mockWatchee));
    EXPECT_CALL(mockWatchee, keepAlive(_)).Times(1);
    EXPECT_CALL(mockWatchee, isTimeout()).WillRepeatedly(Return(false));
    // without wake this will sleep for ever
    EXPECT_CALL(mockWatchee, dueIn(_)).WillRepeatedly(Return(ms(20000)));

    EXPECT_CALL(mockWatchee1, getHandle()).WillRepeatedly(Return(&mockWatchee1));
    EXPECT_CALL(mockWatchee1, keepAlive(_)).Times(1);
    EXPECT_CALL(mockWatchee1, isTimeout()).WillRepeatedly(Return(false));
    EXPECT_CALL(mockWatchee1, dueIn(_)).WillRepeatedly(Return(ms(20000)));

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl));
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl1));
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl));
    ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl1));
}

TEST_F(MVNCWatchdogTests, stressWatchDog) {
    const int num_watchdog_device = 10;
    StrictMock<MockWatchdogDevice> mockWatchee[num_watchdog_device];
    WdDeviceHndl_t deviceHndl[num_watchdog_device];

    for (int i = 0; i != num_watchdog_device; i++) {
        EXPECT_CALL(mockWatchee[i], getHandle()).WillRepeatedly(Return(&mockWatchee[i]));
        // since interval is big keepAlive happens only once
        EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(1);

        EXPECT_CALL(mockWatchee[i], isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mockWatchee[i], dueIn(_)).WillRepeatedly(Return(ms(20000)));

        deviceHndl[i].m_device = &mockWatchee[i];
    }

    for (int k = 0; k != num_watchdog_device; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(MVNCWatchdogTests, stressWatchDog1) {
    const int num_watchdog_device = 10;
    const int num_watchdog_device_half = num_watchdog_device / 2;

    StrictMock<MockWatchdogDevice> mockWatchee[num_watchdog_device];
    WdDeviceHndl_t deviceHndl[num_watchdog_device];

    for (int i = 0; i != num_watchdog_device; i++) {
        EXPECT_CALL(mockWatchee[i], getHandle()).WillRepeatedly(Return(&mockWatchee[i]));
        // since interval is big keepAlive happens only once
        EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(1);

        EXPECT_CALL(mockWatchee[i], isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mockWatchee[i], dueIn(_)).WillRepeatedly(Return(ms(20000)));

        deviceHndl[i].m_device = &mockWatchee[i];
    }

    for (int k = 0; k != num_watchdog_device_half; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device_half; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl[k + num_watchdog_device_half]));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl[k]));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device_half; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl[k + num_watchdog_device_half]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(MVNCWatchdogTests, stressWatchDog2) {
    const int num_watchdog_device = 30;
    const int num_watchdog_device_half1 = num_watchdog_device / 3;
    const int num_watchdog_device_half2 = 2 * num_watchdog_device / 3;

    StrictMock<MockWatchdogDevice> mockWatchee[num_watchdog_device];
    WdDeviceHndl_t deviceHndl[num_watchdog_device];

    for (int i = 0; i != num_watchdog_device; i++) {
        EXPECT_CALL(mockWatchee[i], getHandle()).WillRepeatedly(Return(&mockWatchee[i]));

        // since interval is big keepAlive happens only once
        if (i >= num_watchdog_device_half2) {
            EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(AtLeast(0));
        } else {
            EXPECT_CALL(mockWatchee[i], keepAlive(_)).Times(1);
        }

        EXPECT_CALL(mockWatchee[i], isTimeout()).WillRepeatedly(Return(false));
        EXPECT_CALL(mockWatchee[i], dueIn(_)).WillRepeatedly(Return(ms(20000)));

        deviceHndl[i].m_device = &mockWatchee[i];
    }

    for (int k = 0; k != num_watchdog_device_half1; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = 0; k != num_watchdog_device_half1; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    for (int k = num_watchdog_device_half1; k != num_watchdog_device_half2; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl[k]));
        //this might lead to UB, for example thread might restart but after that device get removed, so giving more time
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    for (int k = num_watchdog_device_half2; k != num_watchdog_device; k++) {
        ASSERT_EQ(WD_ERRNO, watchdog_register_device(m_watchdogHndl,  &deviceHndl[k]));
        //this might lead to UB, for example thread might restart but after that device get removed, so giving more time
        //so our expectations for number of calls are not set for last third
        ASSERT_EQ(WD_ERRNO, watchdog_unregister_device(m_watchdogHndl,  &deviceHndl[k]));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
}
