// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>
#include "mvnc.h"
#include "mvnc_data.h"
#include "ncPrivateTypes.h"
#include "mvnc_common_test_cases.h"

//------------------------------------------------------------------------------
//      MvncTestsCommon Tests
//      Platform independent tests
//------------------------------------------------------------------------------
TEST_F(MvncTestsCommon, DoubleCheckOfAvailableDevicesCount) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    const int min_name_size = 2;

    struct ncDeviceDescr_t act_devices[NC_MAX_DEVICES] = {};
    int act_devicesCount = 0;
    int exp_devicesCount = getAmountOfNotBootedDevices();

    ASSERT_NO_ERROR(ncAvailableDevices(act_devices, NC_MAX_DEVICES, &act_devicesCount));

    ASSERT_TRUE(act_devicesCount);
    ASSERT_EQ(act_devicesCount, exp_devicesCount);

    for (int i = 0; i < act_devicesCount; ++i) {
        ASSERT_GE(strlen(act_devices[i].name), min_name_size);
    }

    for (int j = act_devicesCount; j < NC_MAX_DEVICES; ++j) {
        ASSERT_EQ(strlen(act_devices[j].name), 0);
    }
}

TEST_F(MvncTestsCommon, AvailableDevicesSholdReturnErrorIfArrayIsNULL) {
    int act_devicesCount = 0;
    ASSERT_ERROR(ncAvailableDevices(NULL, NC_MAX_DEVICES, &act_devicesCount));
}

TEST_F(MvncTestsCommon, AvailableDevicesSholdReturnErrorIfCountPtrIsNULL) {
    struct ncDeviceDescr_t act_devices[NC_MAX_DEVICES] = {};
    ASSERT_ERROR(ncAvailableDevices(act_devices, NC_MAX_DEVICES, NULL));
}

TEST_F(MvncTestsCommon, CanGetPCIeAndUSB) {
    if (!(getAmountOfUSBDevices() && getAmountOfPCIeDevices()))
        GTEST_SKIP_("USB and PCIe not available");

    struct ncDeviceDescr_t act_devices[NC_MAX_DEVICES] = {};
    int act_devicesCount = 0;
    ASSERT_NO_ERROR(ncAvailableDevices(act_devices, NC_MAX_DEVICES, &act_devicesCount));

    bool usb_device_found = false;
    bool pcie_device_found = false;

    for (int i = 0; i < act_devicesCount; ++i) {
        if (isMyriadUSBDevice(act_devices[i].name)) {
            usb_device_found = true;
        } else if (isMyriadPCIeDevice(act_devices[i].name)) {
            pcie_device_found = true;
        }
    }

    EXPECT_TRUE(usb_device_found);
    EXPECT_TRUE(pcie_device_found);
}

TEST_F(MvncTestsCommon, ShouldFailToSetNegativeTimeout) {
    ASSERT_ERROR(ncSetDeviceConnectTimeout(-1));
}

//------------------------------------------------------------------------------
//      MvncTestsCommon Tests
//      PCIe + USB Tests
//------------------------------------------------------------------------------

/**
 * @brief Test that USB and PCIe works at the same time. USB first
 */
TEST_F(MvncTestsCommon, OpenUSBThenPCIEAndClose) {
    if (getAmountOfPCIeDevices() == 0)
        GTEST_SKIP() << "PCIe devices not found";
    if (getAmountOfUSBDevices() == 0)
        GTEST_SKIP() << "USB devices not found";

    ncDeviceHandle_t *deviceHandle_USB = nullptr;
    ncDeviceHandle_t *deviceHandle_PCIe = nullptr;
    std::string actDeviceName;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_USB, deviceDesc, m_ncDeviceOpenParams));

    actDeviceName = deviceHandle_USB->private_data->dev_addr;
    ASSERT_TRUE(actDeviceName.size());
    ASSERT_TRUE(isMyriadUSBDevice(actDeviceName));

    // Open PCIe device
    deviceDesc.protocol = NC_PCIE;
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_PCIe, deviceDesc, m_ncDeviceOpenParams));

    actDeviceName = deviceHandle_PCIe->private_data->dev_addr;
    ASSERT_TRUE(actDeviceName.size());
    ASSERT_TRUE(isMyriadPCIeDevice(actDeviceName));

    // Close all
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_PCIe, m_watchdogHndl));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_USB, m_watchdogHndl));
}

/**
 * @brief Test that USB and PCIe works at the same time. PCIe first
 */
TEST_F(MvncTestsCommon, OpenPCIEThenUSBAndClose) {
    if (getAmountOfPCIeDevices() == 0)
        GTEST_SKIP() << "PCIe devices not found";
    if (getAmountOfUSBDevices() == 0)
        GTEST_SKIP() << "USB devices not found";

    ncDeviceHandle_t *deviceHandle_USB = nullptr;
    ncDeviceHandle_t *deviceHandle_PCIe = nullptr;
    std::string actDeviceName;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_PCIE;
    deviceDesc.platform = NC_ANY_PLATFORM;

    // Open PCIe device
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_PCIe, deviceDesc, m_ncDeviceOpenParams));

    actDeviceName = deviceHandle_PCIe->private_data->dev_addr;
    ASSERT_TRUE(actDeviceName.size());
    ASSERT_TRUE(isMyriadPCIeDevice(actDeviceName));

    // Open USB device
    deviceDesc.protocol = NC_USB;
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_USB, deviceDesc, m_ncDeviceOpenParams));

    actDeviceName = deviceHandle_USB->private_data->dev_addr;
    ASSERT_TRUE(actDeviceName.size());
    ASSERT_TRUE(isMyriadUSBDevice(actDeviceName));


    // Close all
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_PCIe, m_watchdogHndl));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_USB, m_watchdogHndl));
}

//------------------------------------------------------------------------------
//      MvncOpenDevice Tests
//------------------------------------------------------------------------------
/**
* @brief Open any device and close it
*/
TEST_P(MvncOpenDevice, OpenAndClose) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    ncDeviceHandle_t*   deviceHandle = nullptr;
    std::string         deviceName;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));

    ASSERT_TRUE(deviceHandle != nullptr);
    ASSERT_TRUE(deviceHandle->private_data != nullptr);
    ASSERT_TRUE(deviceHandle->private_data->dev_addr_booted != nullptr);

    deviceName = deviceHandle->private_data->dev_addr_booted;
    ASSERT_TRUE(deviceName.size() > 0);

    ASSERT_TRUE(isSameProtocolDevice(deviceName, _deviceProtocol));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
}

/**
 * @brief Check that all field of deviceHandle would be initialized
 */
TEST_P(MvncOpenDevice, AllHandleFieldsInitialized) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    ncDeviceHandle_t*   deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));

    ASSERT_TRUE(deviceHandle != nullptr);

    devicePrivate_t * device = deviceHandle->private_data;
    ASSERT_TRUE(device != nullptr);
    ASSERT_TRUE(device->dev_addr != nullptr);
    ASSERT_TRUE(device->dev_addr_booted != nullptr);
    ASSERT_TRUE(device->xlink != nullptr);

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
}

/**
* @brief Try to open device twice. DeviceHandle shouldn't be overwritten
* @details Expected behavior - ncDeviceOpen should warn that deviceHandle
 * already has allocated device
*/
TEST_P(MvncOpenDevice, OpenTwiceSameHandler) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    char dev_addr_first_open[MAX_DEV_NAME];
    unsigned int data_length_first = MAX_DEV_NAME;

    char dev_addr_second_open[MAX_DEV_NAME];
    unsigned int data_length_second = MAX_DEV_NAME;

    // First open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_first_open, &data_length_first));

    // Second open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_second_open, &data_length_second));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
    // Should be the same device
    ASSERT_STREQ(dev_addr_first_open, dev_addr_second_open);
}

/**
 * @brief Try to open device twice with different handlers. Second open should return error
 * @reason #-18548
 */
 // Fixme Test only for one device
TEST_P(MvncOpenDevice, DISABLED_OpenSameDeviceTwiceDifferentHandlers) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    ncDeviceHandle_t *deviceHandle1 = nullptr;
    ncDeviceHandle_t *deviceHandle2 = nullptr;

    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle1, deviceDesc, m_ncDeviceOpenParams));

    // Till we don't have multiple device support, this function would try to open same device
    ASSERT_ERROR(ncDeviceOpen(&deviceHandle2, deviceDesc, m_ncDeviceOpenParams));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle1, m_watchdogHndl));
}


/**
 * @brief Open device twice one run after another. It should check, that link to device closed correctly
 * @note Mostly this test important for PCIe and connect to booted option, as in that cases XLinkReset have another behavior
 */
TEST_P(MvncOpenDevice, OpenTwiceWithOneXLinkInitializion) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    ncDeviceHandle_t *deviceHandle = nullptr;
    std::string actDeviceName;

    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));

    actDeviceName = deviceHandle->private_data->dev_addr;
    ASSERT_TRUE(isSameProtocolDevice(actDeviceName, _deviceProtocol));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));

    // Second open
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));

    actDeviceName = deviceHandle->private_data->dev_addr;
    ASSERT_TRUE(isSameProtocolDevice(actDeviceName, _deviceProtocol));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
}

TEST_P(MvncOpenDevice, WatchdogShouldResetDeviceWithoutConnection) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    std::string         deviceName;
    deviceDesc_t deviceDescToBoot = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = convertProtocolToXlink(_deviceProtocol);
    in_deviceDesc.platform = convertPlatformToXlink(NC_ANY_PLATFORM);
    int expectAvailableDevices = getAmountOfDevices(_deviceProtocol, NC_ANY_PLATFORM, X_LINK_UNBOOTED);

    XLinkError_t rc = X_LINK_ERROR;
    auto waittm = std::chrono::system_clock::now() + std::chrono::seconds(5);
    while ((rc != X_LINK_SUCCESS) && (std::chrono::system_clock::now() < waittm)) {
        rc = XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &deviceDescToBoot);
    }

    bootOptions_t bootOptions = {0};
    bootOptions.wdEnable = 1;

    auto pathToFirmware = getMyriadFirmwarePath(deviceDescToBoot);
    ASSERT_EQ(bootDevice(&deviceDescToBoot, pathToFirmware.c_str(), bootOptions), NC_OK);

    std::this_thread::sleep_for(5_sec);
    ASSERT_EQ(expectAvailableDevices - 1,
        getAmountOfDevices(_deviceProtocol, NC_ANY_PLATFORM, X_LINK_UNBOOTED));

    std::this_thread::sleep_for(15_sec);
    ASSERT_EQ(expectAvailableDevices,
        getAmountOfDevices(_deviceProtocol, NC_ANY_PLATFORM, X_LINK_UNBOOTED));
}

//------------------------------------------------------------------------------
//      MvncLoggingTests Tests
//------------------------------------------------------------------------------
TEST_P(MvncLoggingTests, ShouldNotPrintErrorMessagesIfCanNotOpenDevice) {
    if (availableDevices_ == 0)
        GTEST_SKIP() << ncProtocolToStr(_deviceProtocol) << " devices not found";

    setLogLevel(MVLOG_INFO);
    ncDeviceHandle_t * deviceHandle = nullptr;

    ASSERT_ERROR(ncDeviceOpen(&deviceHandle, _deviceDesc, m_ncDeviceOpenParams));

    std::string content(buff);
    for (int i = MVLOG_WARN; i < MVLOG_LAST; i++) {
        auto found = content.find(mvLogHeader[i]);
        ASSERT_TRUE(found == std::string::npos);
    }
}

//------------------------------------------------------------------------------
//      MvncGraphAllocations Tests
//------------------------------------------------------------------------------
/**
 * @brief Allocate graph for one device
 */
TEST_P(MvncGraphAllocations, DISABLED_OneGraph) {
    if (!blobLoaded) GTEST_SKIP_("Blob for test is not loaded\n");
    openDevices(1, _deviceHandle, _bootedDevices);

    // Create graph handlers
    std::string graphName = "graph";
    ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[0]));
    ASSERT_TRUE(_graphHandle[0] != nullptr);

    // Allocate graph
    ASSERT_NO_ERROR(ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                    _blob.data(), _blob.size(),     // Blob
                                    _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) ));   // Header
}

/**
 * @brief Allocate graphs for 2 device (serial)
 */
TEST_P(MvncGraphAllocations, DISABLED_AllocateGraphsOn2DevicesSerial) {
    if (!blobLoaded)
        GTEST_SKIP_("Blob for test is not loaded\n");
    openDevices(2, _deviceHandle, _bootedDevices);

    // Create graphs handlers
    for (int index = 0; index < _bootedDevices; ++index) {
        std::string graphName = "graph";
        graphName += std::to_string(index);
        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[index]));
        ASSERT_TRUE(_graphHandle[index] != nullptr);
    }

    // Allocate graphs in serial mode
    ncStatus_t rc[MAX_DEVICES];

    for (int i = 0; i < _bootedDevices; ++i) {
        rc[i] = ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                _blob.data(), _blob.size(),     // Blob
                                _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) );  // Header
    }

    for (int i = 0; i < _bootedDevices; ++i) {
        ASSERT_NO_ERROR(rc[i]);
    }
}

/**
* @brief Allocate graphs for 2 device (parallel)
* @detail Open devices and then in parallel threads try to load graphs to it
*         The error easy appear, if USBLINK_TRANSFER_SIZE is (1024 * 1024 * 20)
* @warning It's depend on USBLINK_TRANSFER_SIZE constant from UsbLinkPlatform.c file
* @warning Need blob to use this tests
*/
TEST_P(MvncGraphAllocations, DISABLED_AllocateGraphsOn2DevicesParallel) {
    if (!blobLoaded) GTEST_SKIP_("Blob for test is not loaded\n");
    openDevices(2, _deviceHandle, _bootedDevices);

    // Create graphs handlers
    for (int index = 0; index < _bootedDevices; ++index) {
        std::string graphName = "graph";
        graphName += std::to_string(index);
        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[index]));
        ASSERT_TRUE(_graphHandle[index] != nullptr);
    }

    // Allocate graphs in parallel threads
    std::thread requests[MAX_DEVICES];
    ncStatus_t rc[MAX_DEVICES];
    for (int i = 0; i < _bootedDevices; ++i) {
        requests[i] = std::thread([i, &rc, this]() {
            rc[i] = ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                    _blob.data(), _blob.size(),     // Blob
                                    _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) );
        });
    }

    for (int i = 0; i < _bootedDevices; ++i) {
        requests[i].join();
        ASSERT_NO_ERROR(rc[i]);
    }
}

//------------------------------------------------------------------------------
//      MvncCloseDevice Tests
//------------------------------------------------------------------------------
/**
* @brief Correct closing if handle is empty
*/
TEST_F(MvncCloseDevice, EmptyDeviceHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
}

/**
* @brief Correct closing if some handler fields is null
*/
TEST_F(MvncCloseDevice, EmptyFieldsOfDeviceHandle) {

    ncDeviceHandle_t *deviceHandlePtr;
    auto dH = std::unique_ptr<ncDeviceHandle_t, decltype(std::free)*>(
        (ncDeviceHandle_t*)calloc(1, sizeof(ncDeviceHandle_t)), std::free);

    auto d = std::unique_ptr<_devicePrivate_t,  decltype(std::free)*>(
        (_devicePrivate_t*)calloc(1, sizeof(_devicePrivate_t)), std::free);

    if (dH.get() && d.get()) {
        dH->private_data = d.get();
        d->dev_addr = nullptr;
        d->dev_addr_booted = nullptr;
        d->device_mon_stream_id = INVALID_LINK_ID;
        d->graph_monitor_stream_id = INVALID_LINK_ID;
        d->wd_interval = watchdogInterval;
        deviceHandlePtr = dH.get();
    }

    ASSERT_EQ(ncDeviceClose(&deviceHandlePtr, m_watchdogHndl), NC_INVALID_PARAMETERS);
}

//------------------------------------------------------------------------------
//      MvncInference Tests
//------------------------------------------------------------------------------
using MvncInference = MvncGraphAllocations;

TEST_P(MvncInference, DISABLED_DoOneIterationOfInference) {
    if (!blobLoaded) GTEST_SKIP_("Blob for test is not loaded\n");
    openDevices(1, _deviceHandle, _bootedDevices);

    std::string graphName = "graph";
    ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[0]));
    ASSERT_TRUE(&_graphHandle[0] != nullptr);

    ASSERT_NO_ERROR(ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                    _blob.data(), _blob.size(),     // Blob
                                    _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) ));


    unsigned int dataLength = sizeof(int);

    int numInputs = 0;
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength));

    int numOutputs = 0;
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength));

    dataLength = sizeof(ncTensorDescriptor_t);

    ncTensorDescriptor_t inputDesc = {};
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputDesc,
                                     &dataLength));


    ncTensorDescriptor_t outputDesc = {};
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputDesc,
                                     &dataLength));

    unsigned int fifo_elements = 4;

    ncFifoHandle_t *inputFifoHandle = nullptr;
    ASSERT_NO_ERROR(ncFifoCreate("input", NC_FIFO_HOST_WO, &inputFifoHandle));

    ASSERT_NO_ERROR(ncFifoAllocate(inputFifoHandle, _deviceHandle[0], &inputDesc, fifo_elements));

    ncFifoHandle_t *outputFifoHandle = nullptr;
    ASSERT_NO_ERROR(ncFifoCreate("output", NC_FIFO_HOST_RO, &outputFifoHandle));

    ASSERT_NO_ERROR(ncFifoAllocate(outputFifoHandle, _deviceHandle[0], &outputDesc, fifo_elements));

    uint8_t *input_data = new uint8_t[inputDesc.totalSize];
    uint8_t *result_data = new uint8_t[outputDesc.totalSize];
    ASSERT_NO_ERROR(ncGraphQueueInferenceWithFifoElem(_graphHandle[0],
                                                      inputFifoHandle, outputFifoHandle,
                                                      input_data, &inputDesc.totalSize, nullptr));

    void *userParam = nullptr;
    ASSERT_NO_ERROR(ncFifoReadElem(outputFifoHandle, result_data, &outputDesc.totalSize, &userParam));

    delete[] input_data;
    delete[] result_data;
    ASSERT_NO_ERROR(ncFifoDestroy(&inputFifoHandle));
    ASSERT_NO_ERROR(ncFifoDestroy(&outputFifoHandle));

    ASSERT_NO_ERROR(ncGraphDestroy(&_graphHandle[0]));

    ASSERT_NO_ERROR(ncDeviceClose(&_deviceHandle[0], m_watchdogHndl));
}


INSTANTIATE_TEST_SUITE_P(MvncTestsCommon,
                        MvncOpenDevice,
                        ::testing::ValuesIn(myriadProtocols),
                        PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(MvncTestsCommon,
                        MvncLoggingTests,
                        ::testing::ValuesIn(myriadProtocols),
                        PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(MvncTestsCommon,
                        MvncGraphAllocations,
                        ::testing::ValuesIn(myriadProtocols),
                        PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(MvncTestsCommon,
                        MvncInference,
                        ::testing::ValuesIn(myriadProtocols),
                        PrintToStringParamName());
