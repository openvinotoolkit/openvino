// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>

namespace ov {
namespace intel_myriad {

/**
 * @brief Namespace with Intel HDDL specific properties
 */
namespace hddl {

//    RO properties

/**
 * @brief Property to get a int of the device number
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<int, PropertyMutability::RO> device_num{"HDDL_DEVICE_NUM"};

/**
 * @brief Property to get a std::vector<std::string> of device names
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> device_name{"HDDL_DEVICE_NAME"};

/**
 * @brief  Property to get a std::vector<float> of device thermal
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<float>, PropertyMutability::RO> device_thermal{"HDDL_DEVICE_THERMAL"};

/**
 * @brief  Property to get a std::vector<uint32> of device ids
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<unsigned int>, PropertyMutability::RO> device_id{"HDDL_DEVICE_ID"};

/**
 * @brief  Property to get a std::vector<int> of device subclasses
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<int>, PropertyMutability::RO> device_subclass{"HDDL_DEVICE_SUBCLASS"};

/**
 * @brief  Property to get a std::vector<uint32> of device total memory
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<unsigned int>, PropertyMutability::RO> device_memory_total{
    "HDDL_DEVICE_MEMORY_TOTAL"};

/**
 * @brief  Property to get a std::vector<uint32> of device used memory
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<unsigned int>, PropertyMutability::RO> device_memory_used{
    "HDDL_DEVICE_MEMORY_USED"};

/**
 * @brief  Property to get a std::vector<float> of device utilization
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<float>, PropertyMutability::RO> device_utilization{"HDDL_DEVICE_UTILIZATION"};

/**
 * @brief  Property to get a std::vector<std::string> of stream ids
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> stream_id{"HDDL_STREAM_ID"};

/**
 * @brief  Property to get a std::vector<std::string> of device tags
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> device_tag{"HDDL_DEVICE_TAG"};

/**
 * @brief  Property to get a std::vector<int> of group ids
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::vector<int>, PropertyMutability::RO> group_id{"HDDL_GROUP_ID"};

/**
 * @brief  Property to get a int number of device be using for group
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<int, PropertyMutability::RO> device_group_using_num{"HDDL_DEVICE_GROUP_USING_NUM"};

/**
 * @brief  Property to get a int number of total device
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<int, PropertyMutability::RO> device_total_num{"HDDL_DEVICE_TOTAL_NUM"};

//    RW properties

/**
 * @brief [Only for HDDLPlugin]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This option allows to specify the number of MYX devices used for inference a specific Executable network.
 * Note: Only one network would be allocated to one device.
 * The number of devices for the tag is specified in the hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "graph_tag_map":
 *     {
 *         "tagA":3
 *     }
 * }
 * It means that an executable network marked with tagA will be executed on 3 devices
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RW> graph_tag{"HDDL_GRAPH_TAG"};

/**
 * @brief [Only for HDDLPlugin]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This config makes the executable networks to be allocated on one certain device (instead of multiple devices).
 * And all inference through this executable network, will be done on this device.
 * Note: Only one network would be allocated to one device.
 * The number of devices which will be used for stream-affinity must be specified in hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "stream_device_number":5
 * }
 * It means that 5 device will be used for stream-affinity
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RW> set_stream_id{"HDDL_SET_STREAM_ID"};

/**
 * @brief [Only for HDDLPlugin]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This config allows user to control device flexibly. This config gives a "tag" for a certain device while
 * allocating a network to it. Afterward, user can allocating/deallocating networks to this device with this "tag".
 * Devices used for such use case is controlled by a so-called "Bypass Scheduler" in HDDL backend, and the number
 * of such device need to be specified in hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "bypass_device_number": 5
 * }
 * It means that 5 device will be used for Bypass scheduler.
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RW> set_device_tag{"HDDL_SET_DEVICE_TAG"};

/**
 * @brief [Only for HDDLPlugin]
 * Type: "bool", default is "false".
 * This config is a sub-config of DEVICE_TAG, and only available when "DEVICE_TAG" is set. After a user load a
 * network, the user got a handle for the network.
 * If "YES", the network allocated is bind to the device (with the specified "DEVICE_TAG"), which means all afterwards
 * inference through this network handle will be executed on this device only.
 * If "NO", the network allocated is not bind to the device (with the specified "DEVICE_TAG"). If the same network
 * is allocated on multiple other devices (also set BIND_DEVICE to "False"), then inference through any handle of these
 * networks may be executed on any of these devices those have the network loaded.
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<bool, PropertyMutability::RW> bind_device{"HDDL_BIND_DEVICE"};

/**
 * @brief [Only for HDDLPlugin]
 * Type: A signed int wrapped in a string, default is "0".
 * This config is a sub-config of DEVICE_TAG, and only available when "DEVICE_TAG" is set and "BIND_DEVICE" is "False".
 * When there are multiple devices running a certain network (a same network running on multiple devices in Bypass
 * Scheduler), the device with a larger number has a higher priority, and more inference tasks will be fed to it with
 * priority.
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RW> runtime_priority{"HDDL_RUNTIME_PRIORITY"};

/**
 * @brief [Only for HDDLPlugin]
 * Type: "bool", default is "false".
 * SGAD is short for "Single Graph All Device". With this scheduler, once application allocates 1 network, all devices
 * (managed by SGAD scheduler) will be loaded with this graph. The number of network that can be loaded to one device
 * can exceed one. Once application deallocates 1 network from device, all devices will unload the network from them.
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<bool, PropertyMutability::RW> use_sgad{"HDDL_USE_SGAD"};

/**
 * @brief [Only for HDDLPlugin]
 * Type: A signed int wrapped in a string, default is "0".
 * This config gives a "group id" for a certain device when this device has been reserved for certain client, client
 * can use this device grouped by calling this group id while other client can't use this device
 * Each device has their own group id. Device in one group shares same group id.
 * @ingroup ov_runtime_hddl_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RW> group_device{"HDDL_GROUP_DEVICE"};

}  // namespace hddl
}  // namespace intel_myriad
}  // namespace ov
