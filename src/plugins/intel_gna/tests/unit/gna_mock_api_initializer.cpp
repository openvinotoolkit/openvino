// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_mock_api_initializer.hpp"

#include "gna_mock_api.hpp"
#include <gna2-common-api.h>
#include <cstdint>
#include <vector>

void GnaMockApiInitializer::init() {
    using ::testing::_;
    using ::testing::AtLeast;
    using ::testing::InSequence;
    using ::testing::Invoke;
    using ::testing::Return;

    EXPECT_CALL(_mock_api, Gna2DeviceGetVersion(_, _))
        .WillOnce(Invoke([this](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = this->_gna_device_version;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(_mock_api, Gna2DeviceOpen(_)).WillOnce(Return(Gna2StatusSuccess));

    EXPECT_CALL(_mock_api, Gna2GetLibraryVersion(_, _)).Times(AtLeast(0)).WillRepeatedly(Return(Gna2StatusSuccess));

    EXPECT_CALL(_mock_api, Gna2InstrumentationConfigCreate(_, _, _, _)).WillOnce(Return(Gna2StatusSuccess));

    if (_create_model) {
        EXPECT_CALL(_mock_api, Gna2MemoryAlloc(_, _, _))
            .Times(AtLeast(1))
            .WillRepeatedly(Invoke([this](uint32_t sizeRequested, uint32_t* sizeGranted, void** memoryAddress) {
                this->_mocked_gna_memory.push_back(std::vector<uint8_t>(sizeRequested));
                *sizeGranted = sizeRequested;
                *memoryAddress = this->_mocked_gna_memory.back().data();
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(_mock_api, Gna2ModelCreate(_, _, _))
            .WillOnce(Invoke([](uint32_t deviceIndex, struct Gna2Model const* model, uint32_t* modelId) {
                *modelId = 0;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(_mock_api, Gna2RequestConfigCreate(_, _))
            .WillOnce(Invoke([](uint32_t modelId, uint32_t* requestConfigId) {
                *requestConfigId = 0;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(_mock_api, Gna2InstrumentationConfigAssignToRequestConfig(_, _))
            .Times(AtLeast(1))
            .WillRepeatedly(Return(Gna2StatusSuccess));
    }
    InSequence seq;
    EXPECT_CALL(_mock_api, Gna2DeviceClose(_)).WillOnce(Return(Gna2StatusSuccess));
    if (_create_model) {
        EXPECT_CALL(_mock_api, Gna2MemoryFree(_)).Times(AtLeast(1)).WillRepeatedly(Return(Gna2StatusSuccess));
    }
}
void GnaMockApiInitializer::set_gna_device_version(const Gna2DeviceVersion val) {
    _gna_device_version = val;
}
void GnaMockApiInitializer::set_create_model(const bool val) {
    _create_model = val;
}
