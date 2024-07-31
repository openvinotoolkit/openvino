/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zes_api.h
 * @version v1.8-r1.8.0
 *
 */
#ifndef _ZES_API_H
#define _ZES_API_H
#if defined(__cplusplus)
#pragma once
#endif

// 'core' API headers
#include "ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Level-Zero Sysman API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Handle to a driver instance
typedef ze_driver_handle_t zes_driver_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of device object
typedef ze_device_handle_t zes_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device scheduler queue
typedef struct _zes_sched_handle_t *zes_sched_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device performance factors
typedef struct _zes_perf_handle_t *zes_perf_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device power domain
typedef struct _zes_pwr_handle_t *zes_pwr_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device frequency domain
typedef struct _zes_freq_handle_t *zes_freq_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device engine group
typedef struct _zes_engine_handle_t *zes_engine_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device standby control
typedef struct _zes_standby_handle_t *zes_standby_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device firmware
typedef struct _zes_firmware_handle_t *zes_firmware_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device memory module
typedef struct _zes_mem_handle_t *zes_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman fabric port
typedef struct _zes_fabric_port_handle_t *zes_fabric_port_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device temperature sensor
typedef struct _zes_temp_handle_t *zes_temp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device power supply
typedef struct _zes_psu_handle_t *zes_psu_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device fan
typedef struct _zes_fan_handle_t *zes_fan_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device LED
typedef struct _zes_led_handle_t *zes_led_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device RAS error set
typedef struct _zes_ras_handle_t *zes_ras_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device diagnostics test suite
typedef struct _zes_diag_handle_t *zes_diag_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device overclock domain
typedef struct _zes_overclock_handle_t *zes_overclock_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _zes_structure_type_t
{
    ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1,                             ///< ::zes_device_properties_t
    ZES_STRUCTURE_TYPE_PCI_PROPERTIES = 0x2,                                ///< ::zes_pci_properties_t
    ZES_STRUCTURE_TYPE_PCI_BAR_PROPERTIES = 0x3,                            ///< ::zes_pci_bar_properties_t
    ZES_STRUCTURE_TYPE_DIAG_PROPERTIES = 0x4,                               ///< ::zes_diag_properties_t
    ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES = 0x5,                             ///< ::zes_engine_properties_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_PROPERTIES = 0x6,                        ///< ::zes_fabric_port_properties_t
    ZES_STRUCTURE_TYPE_FAN_PROPERTIES = 0x7,                                ///< ::zes_fan_properties_t
    ZES_STRUCTURE_TYPE_FIRMWARE_PROPERTIES = 0x8,                           ///< ::zes_firmware_properties_t
    ZES_STRUCTURE_TYPE_FREQ_PROPERTIES = 0x9,                               ///< ::zes_freq_properties_t
    ZES_STRUCTURE_TYPE_LED_PROPERTIES = 0xa,                                ///< ::zes_led_properties_t
    ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 0xb,                                ///< ::zes_mem_properties_t
    ZES_STRUCTURE_TYPE_PERF_PROPERTIES = 0xc,                               ///< ::zes_perf_properties_t
    ZES_STRUCTURE_TYPE_POWER_PROPERTIES = 0xd,                              ///< ::zes_power_properties_t
    ZES_STRUCTURE_TYPE_PSU_PROPERTIES = 0xe,                                ///< ::zes_psu_properties_t
    ZES_STRUCTURE_TYPE_RAS_PROPERTIES = 0xf,                                ///< ::zes_ras_properties_t
    ZES_STRUCTURE_TYPE_SCHED_PROPERTIES = 0x10,                             ///< ::zes_sched_properties_t
    ZES_STRUCTURE_TYPE_SCHED_TIMEOUT_PROPERTIES = 0x11,                     ///< ::zes_sched_timeout_properties_t
    ZES_STRUCTURE_TYPE_SCHED_TIMESLICE_PROPERTIES = 0x12,                   ///< ::zes_sched_timeslice_properties_t
    ZES_STRUCTURE_TYPE_STANDBY_PROPERTIES = 0x13,                           ///< ::zes_standby_properties_t
    ZES_STRUCTURE_TYPE_TEMP_PROPERTIES = 0x14,                              ///< ::zes_temp_properties_t
    ZES_STRUCTURE_TYPE_DEVICE_STATE = 0x15,                                 ///< ::zes_device_state_t
    ZES_STRUCTURE_TYPE_PROCESS_STATE = 0x16,                                ///< ::zes_process_state_t
    ZES_STRUCTURE_TYPE_PCI_STATE = 0x17,                                    ///< ::zes_pci_state_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_CONFIG = 0x18,                           ///< ::zes_fabric_port_config_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_STATE = 0x19,                            ///< ::zes_fabric_port_state_t
    ZES_STRUCTURE_TYPE_FAN_CONFIG = 0x1a,                                   ///< ::zes_fan_config_t
    ZES_STRUCTURE_TYPE_FREQ_STATE = 0x1b,                                   ///< ::zes_freq_state_t
    ZES_STRUCTURE_TYPE_OC_CAPABILITIES = 0x1c,                              ///< ::zes_oc_capabilities_t
    ZES_STRUCTURE_TYPE_LED_STATE = 0x1d,                                    ///< ::zes_led_state_t
    ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e,                                    ///< ::zes_mem_state_t
    ZES_STRUCTURE_TYPE_PSU_STATE = 0x1f,                                    ///< ::zes_psu_state_t
    ZES_STRUCTURE_TYPE_BASE_STATE = 0x20,                                   ///< ::zes_base_state_t
    ZES_STRUCTURE_TYPE_RAS_CONFIG = 0x21,                                   ///< ::zes_ras_config_t
    ZES_STRUCTURE_TYPE_RAS_STATE = 0x22,                                    ///< ::zes_ras_state_t
    ZES_STRUCTURE_TYPE_TEMP_CONFIG = 0x23,                                  ///< ::zes_temp_config_t
    ZES_STRUCTURE_TYPE_PCI_BAR_PROPERTIES_1_2 = 0x24,                       ///< ::zes_pci_bar_properties_1_2_t
    ZES_STRUCTURE_TYPE_DEVICE_ECC_DESC = 0x25,                              ///< ::zes_device_ecc_desc_t
    ZES_STRUCTURE_TYPE_DEVICE_ECC_PROPERTIES = 0x26,                        ///< ::zes_device_ecc_properties_t
    ZES_STRUCTURE_TYPE_POWER_LIMIT_EXT_DESC = 0x27,                         ///< ::zes_power_limit_ext_desc_t
    ZES_STRUCTURE_TYPE_POWER_EXT_PROPERTIES = 0x28,                         ///< ::zes_power_ext_properties_t
    ZES_STRUCTURE_TYPE_OVERCLOCK_PROPERTIES = 0x29,                         ///< ::zes_overclock_properties_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_ERROR_COUNTERS = 0x2a,                   ///< ::zes_fabric_port_error_counters_t
    ZES_STRUCTURE_TYPE_ENGINE_EXT_PROPERTIES = 0x2b,                        ///< ::zes_engine_ext_properties_t
    ZES_STRUCTURE_TYPE_RESET_PROPERTIES = 0x2c,                             ///< ::zes_reset_properties_t
    ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES = 0x2d,                        ///< ::zes_device_ext_properties_t
    ZES_STRUCTURE_TYPE_DEVICE_UUID = 0x2e,                                  ///< ::zes_uuid_t
    ZES_STRUCTURE_TYPE_POWER_DOMAIN_EXP_PROPERTIES = 0x00020001,            ///< ::zes_power_domain_exp_properties_t
    ZES_STRUCTURE_TYPE_MEM_TIMESTAMP_BITS_EXP = 0x00020002,                 ///< ::zes_mem_timestamp_bits_exp_t
    ZES_STRUCTURE_TYPE_MEMORY_PAGE_OFFLINE_STATE_EXP = 0x00020003,          ///< ::zes_mem_page_offline_state_exp_t
    ZES_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct _zes_base_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zes_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct _zes_base_desc_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zes_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all state types
typedef struct _zes_base_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zes_base_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all config types
typedef struct _zes_base_config_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zes_base_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all capability types
typedef struct _zes_base_capability_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zes_base_capability_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_properties_t
typedef struct _zes_base_properties_t zes_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_desc_t
typedef struct _zes_base_desc_t zes_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_state_t
typedef struct _zes_base_state_t zes_base_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_config_t
typedef struct _zes_base_config_t zes_base_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_capability_t
typedef struct _zes_base_capability_t zes_base_capability_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_driver_extension_properties_t
typedef struct _zes_driver_extension_properties_t zes_driver_extension_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_state_t
typedef struct _zes_device_state_t zes_device_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_reset_properties_t
typedef struct _zes_reset_properties_t zes_reset_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_uuid_t
typedef struct _zes_uuid_t zes_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_properties_t
typedef struct _zes_device_properties_t zes_device_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_ext_properties_t
typedef struct _zes_device_ext_properties_t zes_device_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_process_state_t
typedef struct _zes_process_state_t zes_process_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_address_t
typedef struct _zes_pci_address_t zes_pci_address_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_speed_t
typedef struct _zes_pci_speed_t zes_pci_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_properties_t
typedef struct _zes_pci_properties_t zes_pci_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_state_t
typedef struct _zes_pci_state_t zes_pci_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_bar_properties_t
typedef struct _zes_pci_bar_properties_t zes_pci_bar_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_bar_properties_1_2_t
typedef struct _zes_pci_bar_properties_1_2_t zes_pci_bar_properties_1_2_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_stats_t
typedef struct _zes_pci_stats_t zes_pci_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_overclock_properties_t
typedef struct _zes_overclock_properties_t zes_overclock_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_control_property_t
typedef struct _zes_control_property_t zes_control_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_vf_property_t
typedef struct _zes_vf_property_t zes_vf_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_diag_test_t
typedef struct _zes_diag_test_t zes_diag_test_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_diag_properties_t
typedef struct _zes_diag_properties_t zes_diag_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_ecc_desc_t
typedef struct _zes_device_ecc_desc_t zes_device_ecc_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_ecc_properties_t
typedef struct _zes_device_ecc_properties_t zes_device_ecc_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_engine_properties_t
typedef struct _zes_engine_properties_t zes_engine_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_engine_stats_t
typedef struct _zes_engine_stats_t zes_engine_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_id_t
typedef struct _zes_fabric_port_id_t zes_fabric_port_id_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_speed_t
typedef struct _zes_fabric_port_speed_t zes_fabric_port_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_properties_t
typedef struct _zes_fabric_port_properties_t zes_fabric_port_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_link_type_t
typedef struct _zes_fabric_link_type_t zes_fabric_link_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_config_t
typedef struct _zes_fabric_port_config_t zes_fabric_port_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_state_t
typedef struct _zes_fabric_port_state_t zes_fabric_port_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_throughput_t
typedef struct _zes_fabric_port_throughput_t zes_fabric_port_throughput_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_error_counters_t
typedef struct _zes_fabric_port_error_counters_t zes_fabric_port_error_counters_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_speed_t
typedef struct _zes_fan_speed_t zes_fan_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_temp_speed_t
typedef struct _zes_fan_temp_speed_t zes_fan_temp_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_speed_table_t
typedef struct _zes_fan_speed_table_t zes_fan_speed_table_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_properties_t
typedef struct _zes_fan_properties_t zes_fan_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_config_t
typedef struct _zes_fan_config_t zes_fan_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_firmware_properties_t
typedef struct _zes_firmware_properties_t zes_firmware_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_properties_t
typedef struct _zes_freq_properties_t zes_freq_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_range_t
typedef struct _zes_freq_range_t zes_freq_range_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_state_t
typedef struct _zes_freq_state_t zes_freq_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_throttle_time_t
typedef struct _zes_freq_throttle_time_t zes_freq_throttle_time_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_oc_capabilities_t
typedef struct _zes_oc_capabilities_t zes_oc_capabilities_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_led_properties_t
typedef struct _zes_led_properties_t zes_led_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_led_color_t
typedef struct _zes_led_color_t zes_led_color_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_led_state_t
typedef struct _zes_led_state_t zes_led_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_properties_t
typedef struct _zes_mem_properties_t zes_mem_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_state_t
typedef struct _zes_mem_state_t zes_mem_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_bandwidth_t
typedef struct _zes_mem_bandwidth_t zes_mem_bandwidth_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_ext_bandwidth_t
typedef struct _zes_mem_ext_bandwidth_t zes_mem_ext_bandwidth_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_perf_properties_t
typedef struct _zes_perf_properties_t zes_perf_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_properties_t
typedef struct _zes_power_properties_t zes_power_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_energy_counter_t
typedef struct _zes_power_energy_counter_t zes_power_energy_counter_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_sustained_limit_t
typedef struct _zes_power_sustained_limit_t zes_power_sustained_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_burst_limit_t
typedef struct _zes_power_burst_limit_t zes_power_burst_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_peak_limit_t
typedef struct _zes_power_peak_limit_t zes_power_peak_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_energy_threshold_t
typedef struct _zes_energy_threshold_t zes_energy_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_psu_properties_t
typedef struct _zes_psu_properties_t zes_psu_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_psu_state_t
typedef struct _zes_psu_state_t zes_psu_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_properties_t
typedef struct _zes_ras_properties_t zes_ras_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_state_t
typedef struct _zes_ras_state_t zes_ras_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_config_t
typedef struct _zes_ras_config_t zes_ras_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_sched_properties_t
typedef struct _zes_sched_properties_t zes_sched_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_sched_timeout_properties_t
typedef struct _zes_sched_timeout_properties_t zes_sched_timeout_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_sched_timeslice_properties_t
typedef struct _zes_sched_timeslice_properties_t zes_sched_timeslice_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_standby_properties_t
typedef struct _zes_standby_properties_t zes_standby_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_temp_properties_t
typedef struct _zes_temp_properties_t zes_temp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_temp_threshold_t
typedef struct _zes_temp_threshold_t zes_temp_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_temp_config_t
typedef struct _zes_temp_config_t zes_temp_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_limit_ext_desc_t
typedef struct _zes_power_limit_ext_desc_t zes_power_limit_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_ext_properties_t
typedef struct _zes_power_ext_properties_t zes_power_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_engine_ext_properties_t
typedef struct _zes_engine_ext_properties_t zes_engine_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_state_exp_t
typedef struct _zes_ras_state_exp_t zes_ras_state_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_page_offline_state_exp_t
typedef struct _zes_mem_page_offline_state_exp_t zes_mem_page_offline_state_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_timestamp_bits_exp_t
typedef struct _zes_mem_timestamp_bits_exp_t zes_mem_timestamp_bits_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_domain_exp_properties_t
typedef struct _zes_power_domain_exp_properties_t zes_power_domain_exp_properties_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman)
#if !defined(__GNUC__)
#pragma region driver
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported sysman initialization flags
typedef uint32_t zes_init_flags_t;
typedef enum _zes_init_flag_t
{
    ZES_INIT_FLAG_PLACEHOLDER = ZE_BIT(0),                                  ///< placeholder for future use
    ZES_INIT_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_init_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Initialize 'oneAPI' System Resource Management (sysman)
/// 
/// @details
///     - The application must call this function or ::zeInit with the
///       ::ZES_ENABLE_SYSMAN environment variable set before calling any other
///       sysman function.
///     - If this function is not called then all other sysman functions will
///       return ::ZE_RESULT_ERROR_UNINITIALIZED.
///     - This function will only initialize sysman. To initialize other
///       functions, call ::zeInit.
///     - There is no requirement to call this function before or after
///       ::zeInit.
///     - Only one instance of sysman will be initialized per process.
///     - The application must call this function after forking new processes.
///       Each forked process must call this function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe for scenarios
///       where multiple libraries may initialize sysman simultaneously.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < flags`
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
ZE_APIEXPORT ze_result_t ZE_APICALL
zesInit(
    zes_init_flags_t flags                                                  ///< [in] initialization flags.
                                                                            ///< currently unused, must be 0 (default).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves sysman driver instances
/// 
/// @details
///     - A sysman driver represents a collection of physical devices.
///     - Multiple calls to this function will return identical sysman driver
///       handles, in the same order.
///     - The application may pass nullptr for pDrivers when only querying the
///       number of sysman drivers.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverGet(
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of sysman driver instances.
                                                                            ///< if count is zero, then the loader shall update the value with the
                                                                            ///< total number of sysman drivers available.
                                                                            ///< if count is greater than the number of sysman drivers available, then
                                                                            ///< the loader shall update the value with the correct number of sysman
                                                                            ///< drivers available.
    zes_driver_handle_t* phDrivers                                          ///< [in,out][optional][range(0, *pCount)] array of sysman driver instance handles.
                                                                            ///< if count is less than the number of sysman drivers available, then the
                                                                            ///< loader shall only retrieve that number of sysman drivers.
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_EXTENSION_NAME
/// @brief Maximum extension name string size
#define ZES_MAX_EXTENSION_NAME  256
#endif // ZES_MAX_EXTENSION_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties queried using ::zesDriverGetExtensionProperties
typedef struct _zes_driver_extension_properties_t
{
    char name[ZES_MAX_EXTENSION_NAME];                                      ///< [out] extension name
    uint32_t version;                                                       ///< [out] extension version using ::ZE_MAKE_VERSION

} zes_driver_extension_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves extension properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverGetExtensionProperties(
    zes_driver_handle_t hDriver,                                            ///< [in] handle of the driver instance
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of extension properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of extension properties available.
                                                                            ///< if count is greater than the number of extension properties available,
                                                                            ///< then the driver shall update the value with the correct number of
                                                                            ///< extension properties available.
    zes_driver_extension_properties_t* pExtensionProperties                 ///< [in,out][optional][range(0, *pCount)] array of query results for
                                                                            ///< extension properties.
                                                                            ///< if count is less than the number of extension properties available,
                                                                            ///< then driver shall only retrieve that number of extension properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves function pointer for vendor-specific or experimental
///        extensions
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == name`
///         + `nullptr == ppFunctionAddress`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverGetExtensionFunctionAddress(
    zes_driver_handle_t hDriver,                                            ///< [in] handle of the driver instance
    const char* name,                                                       ///< [in] extension function name
    void** ppFunctionAddress                                                ///< [out] pointer to function pointer
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Device management
#if !defined(__GNUC__)
#pragma region device
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves sysman devices within a sysman driver
/// 
/// @details
///     - Multiple calls to this function will return identical sysman device
///       handles, in the same order.
///     - The number and order of handles returned from this function is NOT
///       affected by the ::ZE_AFFINITY_MASK, ::ZE_ENABLE_PCI_ID_DEVICE_ORDER,
///       or ::ZE_FLAT_DEVICE_HIERARCHY environment variables.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGet(
    zes_driver_handle_t hDriver,                                            ///< [in] handle of the sysman driver instance
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of sysman devices.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of sysman devices available.
                                                                            ///< if count is greater than the number of sysman devices available, then
                                                                            ///< the driver shall update the value with the correct number of sysman
                                                                            ///< devices available.
    zes_device_handle_t* phDevices                                          ///< [in,out][optional][range(0, *pCount)] array of handle of sysman devices.
                                                                            ///< if count is less than the number of sysman devices available, then
                                                                            ///< driver shall only retrieve that number of sysman devices.
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_STRING_PROPERTY_SIZE
/// @brief Maximum number of characters in string properties.
#define ZES_STRING_PROPERTY_SIZE  64
#endif // ZES_STRING_PROPERTY_SIZE

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_UUID_SIZE
/// @brief Maximum device universal unique id (UUID) size in bytes.
#define ZES_MAX_UUID_SIZE  16
#endif // ZES_MAX_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Types of accelerator engines
typedef uint32_t zes_engine_type_flags_t;
typedef enum _zes_engine_type_flag_t
{
    ZES_ENGINE_TYPE_FLAG_OTHER = ZE_BIT(0),                                 ///< Undefined types of accelerators.
    ZES_ENGINE_TYPE_FLAG_COMPUTE = ZE_BIT(1),                               ///< Engines that process compute kernels only (no 3D content).
    ZES_ENGINE_TYPE_FLAG_3D = ZE_BIT(2),                                    ///< Engines that process 3D content only (no compute kernels).
    ZES_ENGINE_TYPE_FLAG_MEDIA = ZE_BIT(3),                                 ///< Engines that process media workloads.
    ZES_ENGINE_TYPE_FLAG_DMA = ZE_BIT(4),                                   ///< Engines that copy blocks of data.
    ZES_ENGINE_TYPE_FLAG_RENDER = ZE_BIT(5),                                ///< Engines that can process both 3D content and compute kernels.
    ZES_ENGINE_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_engine_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device repair status
typedef enum _zes_repair_status_t
{
    ZES_REPAIR_STATUS_UNSUPPORTED = 0,                                      ///< The device does not support in-field repairs.
    ZES_REPAIR_STATUS_NOT_PERFORMED = 1,                                    ///< The device has never been repaired.
    ZES_REPAIR_STATUS_PERFORMED = 2,                                        ///< The device has been repaired.
    ZES_REPAIR_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_repair_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device reset reasons
typedef uint32_t zes_reset_reason_flags_t;
typedef enum _zes_reset_reason_flag_t
{
    ZES_RESET_REASON_FLAG_WEDGED = ZE_BIT(0),                               ///< The device needs to be reset because one or more parts of the hardware
                                                                            ///< is wedged
    ZES_RESET_REASON_FLAG_REPAIR = ZE_BIT(1),                               ///< The device needs to be reset in order to complete in-field repairs
    ZES_RESET_REASON_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_reset_reason_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device reset type
typedef enum _zes_reset_type_t
{
    ZES_RESET_TYPE_WARM = 0,                                                ///< Apply warm reset
    ZES_RESET_TYPE_COLD = 1,                                                ///< Apply cold reset
    ZES_RESET_TYPE_FLR = 2,                                                 ///< Apply FLR reset
    ZES_RESET_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_reset_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device state
typedef struct _zes_device_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_reset_reason_flags_t reset;                                         ///< [out] Indicates if the device needs to be reset and for what reasons.
                                                                            ///< returns 0 (none) or combination of ::zes_reset_reason_flag_t
    zes_repair_status_t repaired;                                           ///< [out] Indicates if the device has been repaired

} zes_device_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device reset properties
typedef struct _zes_reset_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t force;                                                        ///< [in] If set to true, all applications that are currently using the
                                                                            ///< device will be forcibly killed.
    zes_reset_type_t resetType;                                             ///< [in] Type of reset needs to be performed

} zes_reset_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device universal unique id (UUID)
typedef struct _zes_uuid_t
{
    uint8_t id[ZES_MAX_UUID_SIZE];                                          ///< [out] opaque data representing a device UUID

} zes_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum _zes_device_type_t
{
    ZES_DEVICE_TYPE_GPU = 1,                                                ///< Graphics Processing Unit
    ZES_DEVICE_TYPE_CPU = 2,                                                ///< Central Processing Unit
    ZES_DEVICE_TYPE_FPGA = 3,                                               ///< Field Programmable Gate Array
    ZES_DEVICE_TYPE_MCA = 4,                                                ///< Memory Copy Accelerator
    ZES_DEVICE_TYPE_VPU = 5,                                                ///< Vision Processing Unit
    ZES_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_device_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device property flags
typedef uint32_t zes_device_property_flags_t;
typedef enum _zes_device_property_flag_t
{
    ZES_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0),                        ///< Device is integrated with the Host.
    ZES_DEVICE_PROPERTY_FLAG_SUBDEVICE = ZE_BIT(1),                         ///< Device handle used for query represents a sub-device.
    ZES_DEVICE_PROPERTY_FLAG_ECC = ZE_BIT(2),                               ///< Device supports error correction memory access.
    ZES_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = ZE_BIT(3),                    ///< Device supports on-demand page-faulting.
    ZES_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_device_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device properties
typedef struct _zes_device_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_properties_t core;                                            ///< [out] (Deprecated, use ::zes_uuid_t in the extended structure) Core
                                                                            ///< device properties
    uint32_t numSubdevices;                                                 ///< [out] Number of sub-devices. A value of 0 indicates that this device
                                                                            ///< doesn't have sub-devices.
    char serialNumber[ZES_STRING_PROPERTY_SIZE];                            ///< [out] Manufacturing serial number (NULL terminated string value). This
                                                                            ///< value is intended to reflect the Part ID/SoC ID assigned by
                                                                            ///< manufacturer that is unique for a SoC. Will be set to the string
                                                                            ///< "unknown" if this cannot be determined for the device.
    char boardNumber[ZES_STRING_PROPERTY_SIZE];                             ///< [out] Manufacturing board number (NULL terminated string value).
                                                                            ///< Alternatively "boardSerialNumber", this value is intended to reflect
                                                                            ///< the string printed on board label by manufacturer. Will be set to the
                                                                            ///< string "unknown" if this cannot be determined for the device.
    char brandName[ZES_STRING_PROPERTY_SIZE];                               ///< [out] Brand name of the device (NULL terminated string value). Will be
                                                                            ///< set to the string "unknown" if this cannot be determined for the
                                                                            ///< device.
    char modelName[ZES_STRING_PROPERTY_SIZE];                               ///< [out] Model name of the device (NULL terminated string value). Will be
                                                                            ///< set to the string "unknown" if this cannot be determined for the
                                                                            ///< device.
    char vendorName[ZES_STRING_PROPERTY_SIZE];                              ///< [out] Vendor name of the device (NULL terminated string value). Will
                                                                            ///< be set to the string "unknown" if this cannot be determined for the
                                                                            ///< device.
    char driverVersion[ZES_STRING_PROPERTY_SIZE];                           ///< [out] Installed driver version (NULL terminated string value). Will be
                                                                            ///< set to the string "unknown" if this cannot be determined for the
                                                                            ///< device.

} zes_device_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device properties
typedef struct _zes_device_ext_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_uuid_t uuid;                                                        ///< [out] universal unique identifier. Note: uuid obtained from Sysman API
                                                                            ///< is the same as from core API. Subdevices will have their own uuid.
    zes_device_type_t type;                                                 ///< [out] generic device type
    zes_device_property_flags_t flags;                                      ///< [out] 0 (none) or a valid combination of ::zes_device_property_flag_t

} zes_device_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties about the device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetProperties(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_device_properties_t* pProperties                                    ///< [in,out] Structure that will contain information about the device.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about the state of the device - if a reset is
///        required, reasons for the reset and if the device has been repaired
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetState(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_device_state_t* pState                                              ///< [in,out] Structure that will contain information about the device.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reset device
/// 
/// @details
///     - Performs a PCI bus reset of the device. This will result in all
///       current device state being lost.
///     - All applications using the device should be stopped before calling
///       this function.
///     - If the force argument is specified, all applications using the device
///       will be forcibly killed.
///     - The function will block until the device has restarted or an
///       implementation defined timeout occurred waiting for the reset to
///       complete.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform this operation.
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + Reset cannot be performed because applications are using this device.
///     - ::ZE_RESULT_ERROR_UNKNOWN
///         + There were problems unloading the device driver, performing a bus reset or reloading the device driver.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceReset(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle for the device
    ze_bool_t force                                                         ///< [in] If set to true, all applications that are currently using the
                                                                            ///< device will be forcibly killed.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reset device extension
/// 
/// @details
///     - Performs a PCI bus reset of the device. This will result in all
///       current device state being lost.
///     - Prior to calling this function, user is responsible for closing
///       applications using the device unless force argument is specified.
///     - If the force argument is specified, all applications using the device
///       will be forcibly killed.
///     - The function will block until the device has restarted or a
///       implementation specific timeout occurred waiting for the reset to
///       complete.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform this operation.
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + Reset cannot be performed because applications are using this device.
///     - ::ZE_RESULT_ERROR_UNKNOWN
///         + There were problems unloading the device driver, performing a bus reset or reloading the device driver.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceResetExt(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle for the device
    zes_reset_properties_t* pProperties                                     ///< [in] Device reset properties to apply
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Contains information about a process that has an open connection with
///        this device
/// 
/// @details
///     - The application can use the process ID to query the OS for the owner
///       and the path to the executable.
typedef struct _zes_process_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t processId;                                                     ///< [out] Host OS process ID.
    uint64_t memSize;                                                       ///< [out] Device memory size in bytes allocated by this process (may not
                                                                            ///< necessarily be resident on the device at the time of reading).
    uint64_t sharedSize;                                                    ///< [out] The size of shared device memory mapped into this process (may
                                                                            ///< not necessarily be resident on the device at the time of reading).
    zes_engine_type_flags_t engines;                                        ///< [out] Bitfield of accelerator engine types being used by this process.

} zes_process_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about host processes using the device
/// 
/// @details
///     - The number of processes connected to the device is dynamic. This means
///       that between a call to determine the value of pCount and the
///       subsequent call, the number of processes may have increased or
///       decreased. It is recommended that a large array be passed in so as to
///       avoid receiving the error ::ZE_RESULT_ERROR_INVALID_SIZE. Also, always
///       check the returned value in pCount since it may be less than the
///       earlier call to get the required array size.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + The provided value of pCount is not big enough to store information about all the processes currently attached to the device.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceProcessesGetState(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle for the device
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of processes.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of processes currently attached to the device.
                                                                            ///< if count is greater than the number of processes currently attached to
                                                                            ///< the device, then the driver shall update the value with the correct
                                                                            ///< number of processes.
    zes_process_state_t* pProcesses                                         ///< [in,out][optional][range(0, *pCount)] array of process information.
                                                                            ///< if count is less than the number of processes currently attached to
                                                                            ///< the device, then the driver shall only retrieve information about that
                                                                            ///< number of processes. In this case, the return code will ::ZE_RESULT_ERROR_INVALID_SIZE.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI address
typedef struct _zes_pci_address_t
{
    uint32_t domain;                                                        ///< [out] BDF domain
    uint32_t bus;                                                           ///< [out] BDF bus
    uint32_t device;                                                        ///< [out] BDF device
    uint32_t function;                                                      ///< [out] BDF function

} zes_pci_address_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI speed
typedef struct _zes_pci_speed_t
{
    int32_t gen;                                                            ///< [out] The link generation. A value of -1 means that this property is
                                                                            ///< unknown.
    int32_t width;                                                          ///< [out] The number of lanes. A value of -1 means that this property is
                                                                            ///< unknown.
    int64_t maxBandwidth;                                                   ///< [out] The maximum bandwidth in bytes/sec (sum of all lanes). A value
                                                                            ///< of -1 means that this property is unknown.

} zes_pci_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Static PCI properties
typedef struct _zes_pci_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_pci_address_t address;                                              ///< [out] The BDF address
    zes_pci_speed_t maxSpeed;                                               ///< [out] Fastest port configuration supported by the device (sum of all
                                                                            ///< lanes)
    ze_bool_t haveBandwidthCounters;                                        ///< [out] Indicates whether the `rxCounter` and `txCounter` members of
                                                                            ///< ::zes_pci_stats_t will have valid values
    ze_bool_t havePacketCounters;                                           ///< [out] Indicates whether the `packetCounter` member of
                                                                            ///< ::zes_pci_stats_t will have a valid value
    ze_bool_t haveReplayCounters;                                           ///< [out] Indicates whether the `replayCounter` member of
                                                                            ///< ::zes_pci_stats_t will have a valid value

} zes_pci_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI link status
typedef enum _zes_pci_link_status_t
{
    ZES_PCI_LINK_STATUS_UNKNOWN = 0,                                        ///< The link status could not be determined
    ZES_PCI_LINK_STATUS_GOOD = 1,                                           ///< The link is up and operating as expected
    ZES_PCI_LINK_STATUS_QUALITY_ISSUES = 2,                                 ///< The link is up but has quality and/or bandwidth degradation
    ZES_PCI_LINK_STATUS_STABILITY_ISSUES = 3,                               ///< The link has stability issues and preventing workloads making forward
                                                                            ///< progress
    ZES_PCI_LINK_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI link quality degradation reasons
typedef uint32_t zes_pci_link_qual_issue_flags_t;
typedef enum _zes_pci_link_qual_issue_flag_t
{
    ZES_PCI_LINK_QUAL_ISSUE_FLAG_REPLAYS = ZE_BIT(0),                       ///< A significant number of replays are occurring
    ZES_PCI_LINK_QUAL_ISSUE_FLAG_SPEED = ZE_BIT(1),                         ///< There is a degradation in the maximum bandwidth of the link
    ZES_PCI_LINK_QUAL_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_qual_issue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI link stability issues
typedef uint32_t zes_pci_link_stab_issue_flags_t;
typedef enum _zes_pci_link_stab_issue_flag_t
{
    ZES_PCI_LINK_STAB_ISSUE_FLAG_RETRAINING = ZE_BIT(0),                    ///< Link retraining has occurred to deal with quality issues
    ZES_PCI_LINK_STAB_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_stab_issue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Dynamic PCI state
typedef struct _zes_pci_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_pci_link_status_t status;                                           ///< [out] The current status of the port
    zes_pci_link_qual_issue_flags_t qualityIssues;                          ///< [out] If status is ::ZES_PCI_LINK_STATUS_QUALITY_ISSUES, 
                                                                            ///< then this gives a combination of ::zes_pci_link_qual_issue_flag_t for
                                                                            ///< quality issues that have been detected;
                                                                            ///< otherwise, 0 indicates there are no quality issues with the link at
                                                                            ///< this time."
    zes_pci_link_stab_issue_flags_t stabilityIssues;                        ///< [out] If status is ::ZES_PCI_LINK_STATUS_STABILITY_ISSUES, 
                                                                            ///< then this gives a combination of ::zes_pci_link_stab_issue_flag_t for
                                                                            ///< reasons for the connection instability;
                                                                            ///< otherwise, 0 indicates there are no connection stability issues at
                                                                            ///< this time."
    zes_pci_speed_t speed;                                                  ///< [out] The current port configure speed

} zes_pci_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI bar types
typedef enum _zes_pci_bar_type_t
{
    ZES_PCI_BAR_TYPE_MMIO = 0,                                              ///< MMIO registers
    ZES_PCI_BAR_TYPE_ROM = 1,                                               ///< ROM aperture
    ZES_PCI_BAR_TYPE_MEM = 2,                                               ///< Device memory
    ZES_PCI_BAR_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_pci_bar_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties of a pci bar
typedef struct _zes_pci_bar_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_pci_bar_type_t type;                                                ///< [out] The type of bar
    uint32_t index;                                                         ///< [out] The index of the bar
    uint64_t base;                                                          ///< [out] Base address of the bar.
    uint64_t size;                                                          ///< [out] Size of the bar.

} zes_pci_bar_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties of a pci bar, including the resizable bar.
typedef struct _zes_pci_bar_properties_1_2_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_pci_bar_type_t type;                                                ///< [out] The type of bar
    uint32_t index;                                                         ///< [out] The index of the bar
    uint64_t base;                                                          ///< [out] Base address of the bar.
    uint64_t size;                                                          ///< [out] Size of the bar.
    ze_bool_t resizableBarSupported;                                        ///< [out] Support for Resizable Bar on this device.
    ze_bool_t resizableBarEnabled;                                          ///< [out] Resizable Bar enabled on this device

} zes_pci_bar_properties_1_2_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI stats counters
/// 
/// @details
///     - Percent replays is calculated by taking two snapshots (s1, s2) and
///       using the equation: %replay = 10^6 * (s2.replayCounter -
///       s1.replayCounter) / (s2.maxBandwidth * (s2.timestamp - s1.timestamp))
///     - Percent throughput is calculated by taking two snapshots (s1, s2) and
///       using the equation: %bw = 10^6 * ((s2.rxCounter - s1.rxCounter) +
///       (s2.txCounter - s1.txCounter)) / (s2.maxBandwidth * (s2.timestamp -
///       s1.timestamp))
typedef struct _zes_pci_stats_t
{
    uint64_t timestamp;                                                     ///< [out] Monotonic timestamp counter in microseconds when the measurement
                                                                            ///< was made.
                                                                            ///< This timestamp should only be used to calculate delta time between
                                                                            ///< snapshots of this structure.
                                                                            ///< Never take the delta of this timestamp with the timestamp from a
                                                                            ///< different structure since they are not guaranteed to have the same base.
                                                                            ///< The absolute value of the timestamp is only valid during within the
                                                                            ///< application and may be different on the next execution.
    uint64_t replayCounter;                                                 ///< [out] Monotonic counter for the number of replay packets (sum of all
                                                                            ///< lanes). Will always be 0 when the `haveReplayCounters` member of
                                                                            ///< ::zes_pci_properties_t is FALSE.
    uint64_t packetCounter;                                                 ///< [out] Monotonic counter for the number of packets (sum of all lanes).
                                                                            ///< Will always be 0 when the `havePacketCounters` member of
                                                                            ///< ::zes_pci_properties_t is FALSE.
    uint64_t rxCounter;                                                     ///< [out] Monotonic counter for the number of bytes received (sum of all
                                                                            ///< lanes). Will always be 0 when the `haveBandwidthCounters` member of
                                                                            ///< ::zes_pci_properties_t is FALSE.
    uint64_t txCounter;                                                     ///< [out] Monotonic counter for the number of bytes transmitted (including
                                                                            ///< replays) (sum of all lanes). Will always be 0 when the
                                                                            ///< `haveBandwidthCounters` member of ::zes_pci_properties_t is FALSE.
    zes_pci_speed_t speed;                                                  ///< [out] The current speed of the link (sum of all lanes)

} zes_pci_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get PCI properties - address, max speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetProperties(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_pci_properties_t* pProperties                                       ///< [in,out] Will contain the PCI properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current PCI state - current speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetState(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_pci_state_t* pState                                                 ///< [in,out] Will contain the PCI properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about each configured bar
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetBars(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of PCI bars.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of PCI bars that are setup.
                                                                            ///< if count is greater than the number of PCI bars that are setup, then
                                                                            ///< the driver shall update the value with the correct number of PCI bars.
    zes_pci_bar_properties_t* pProperties                                   ///< [in,out][optional][range(0, *pCount)] array of information about setup
                                                                            ///< PCI bars.
                                                                            ///< if count is less than the number of PCI bars that are setup, then the
                                                                            ///< driver shall only retrieve information about that number of PCI bars.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get PCI stats - bandwidth, number of packets, number of replays
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pStats`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetStats(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_pci_stats_t* pStats                                                 ///< [in,out] Will contain a snapshot of the latest stats.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Overclock controls, VF curve manipulation
#if !defined(__GNUC__)
#pragma region Overclock
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock domains.
typedef enum _zes_overclock_domain_t
{
    ZES_OVERCLOCK_DOMAIN_CARD = 1,                                          ///< Overclocking card level properties such as temperature limits.
    ZES_OVERCLOCK_DOMAIN_PACKAGE = 2,                                       ///< Overclocking package level properties such as power limits.
    ZES_OVERCLOCK_DOMAIN_GPU_ALL = 4,                                       ///< Overclocking a GPU that has all accelerator assets on the same PLL/VR.
    ZES_OVERCLOCK_DOMAIN_GPU_RENDER_COMPUTE = 8,                            ///< Overclocking a GPU with render and compute assets on the same PLL/VR.
    ZES_OVERCLOCK_DOMAIN_GPU_RENDER = 16,                                   ///< Overclocking a GPU with render assets on its own PLL/VR.
    ZES_OVERCLOCK_DOMAIN_GPU_COMPUTE = 32,                                  ///< Overclocking a GPU with compute assets on its own PLL/VR.
    ZES_OVERCLOCK_DOMAIN_GPU_MEDIA = 64,                                    ///< Overclocking a GPU with media assets on its own PLL/VR.
    ZES_OVERCLOCK_DOMAIN_VRAM = 128,                                        ///< Overclocking device local memory.
    ZES_OVERCLOCK_DOMAIN_ADM = 256,                                         ///< Overclocking LLC/L4 cache.
    ZES_OVERCLOCK_DOMAIN_FORCE_UINT32 = 0x7fffffff

} zes_overclock_domain_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock controls.
typedef enum _zes_overclock_control_t
{
    ZES_OVERCLOCK_CONTROL_VF = 1,                                           ///< This control permits setting a custom V-F curve.
    ZES_OVERCLOCK_CONTROL_FREQ_OFFSET = 2,                                  ///< The V-F curve of the overclock domain can be shifted up or down using
                                                                            ///< this control.
    ZES_OVERCLOCK_CONTROL_VMAX_OFFSET = 4,                                  ///< This control is used to increase the permitted voltage above the
                                                                            ///< shipped voltage maximum.
    ZES_OVERCLOCK_CONTROL_FREQ = 8,                                         ///< This control permits direct changes to the operating frequency.
    ZES_OVERCLOCK_CONTROL_VOLT_LIMIT = 16,                                  ///< This control prevents frequencies that would push the voltage above
                                                                            ///< this value, typically used by V-F scanners.
    ZES_OVERCLOCK_CONTROL_POWER_SUSTAINED_LIMIT = 32,                       ///< This control changes the sustained power limit (PL1).
    ZES_OVERCLOCK_CONTROL_POWER_BURST_LIMIT = 64,                           ///< This control changes the burst power limit (PL2).
    ZES_OVERCLOCK_CONTROL_POWER_PEAK_LIMIT = 128,                           ///< his control changes the peak power limit (PL4).
    ZES_OVERCLOCK_CONTROL_ICCMAX_LIMIT = 256,                               ///< This control changes the value of IccMax..
    ZES_OVERCLOCK_CONTROL_TEMP_LIMIT = 512,                                 ///< This control changes the value of TjMax.
    ZES_OVERCLOCK_CONTROL_ITD_DISABLE = 1024,                               ///< This control permits disabling the adaptive voltage feature ITD
    ZES_OVERCLOCK_CONTROL_ACM_DISABLE = 2048,                               ///< This control permits disabling the adaptive voltage feature ACM.
    ZES_OVERCLOCK_CONTROL_FORCE_UINT32 = 0x7fffffff

} zes_overclock_control_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock modes.
typedef enum _zes_overclock_mode_t
{
    ZES_OVERCLOCK_MODE_MODE_OFF = 0,                                        ///< Overclock mode is off
    ZES_OVERCLOCK_MODE_MODE_STOCK = 2,                                      ///< Stock (manufacturing settings) are being used.
    ZES_OVERCLOCK_MODE_MODE_ON = 3,                                         ///< Overclock mode is on.
    ZES_OVERCLOCK_MODE_MODE_UNAVAILABLE = 4,                                ///< Overclocking is unavailable at this time since the system is running
                                                                            ///< on battery.
    ZES_OVERCLOCK_MODE_MODE_DISABLED = 5,                                   ///< Overclock mode is disabled.
    ZES_OVERCLOCK_MODE_FORCE_UINT32 = 0x7fffffff

} zes_overclock_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock control states.
typedef enum _zes_control_state_t
{
    ZES_CONTROL_STATE_STATE_UNSET = 0,                                      ///< No overclock control has not been changed by the driver since the last
                                                                            ///< boot/reset.
    ZES_CONTROL_STATE_STATE_ACTIVE = 2,                                     ///< The overclock control has been set and it is active.
    ZES_CONTROL_STATE_STATE_DISABLED = 3,                                   ///< The overclock control value has been disabled due to the current power
                                                                            ///< configuration (typically when running on DC).
    ZES_CONTROL_STATE_FORCE_UINT32 = 0x7fffffff

} zes_control_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock pending actions.
typedef enum _zes_pending_action_t
{
    ZES_PENDING_ACTION_PENDING_NONE = 0,                                    ///< There no pending actions. .
    ZES_PENDING_ACTION_PENDING_IMMINENT = 1,                                ///< The requested change is in progress and should complete soon.
    ZES_PENDING_ACTION_PENDING_COLD_RESET = 2,                              ///< The requested change requires a device cold reset (hotplug, system
                                                                            ///< boot).
    ZES_PENDING_ACTION_PENDING_WARM_RESET = 3,                              ///< The requested change requires a device warm reset (PCIe FLR).
    ZES_PENDING_ACTION_FORCE_UINT32 = 0x7fffffff

} zes_pending_action_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock V-F curve programing.
typedef enum _zes_vf_program_type_t
{
    ZES_VF_PROGRAM_TYPE_VF_ARBITRARY = 0,                                   ///< Can program an arbitrary number of V-F points up to the maximum number
                                                                            ///< and each point can have arbitrary voltage and frequency values within
                                                                            ///< the min/max/step limits
    ZES_VF_PROGRAM_TYPE_VF_FREQ_FIXED = 1,                                  ///< Can only program the voltage for the V-F points that it reads back -
                                                                            ///< the frequency of those points cannot be changed
    ZES_VF_PROGRAM_TYPE_VF_VOLT_FIXED = 2,                                  ///< Can only program the frequency for the V-F points that is reads back -
                                                                            ///< the voltage of each point cannot be changed.
    ZES_VF_PROGRAM_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_vf_program_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief VF type
typedef enum _zes_vf_type_t
{
    ZES_VF_TYPE_VOLT = 0,                                                   ///< VF Voltage point
    ZES_VF_TYPE_FREQ = 1,                                                   ///< VF Frequency point
    ZES_VF_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_vf_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief VF type
typedef enum _zes_vf_array_type_t
{
    ZES_VF_ARRAY_TYPE_USER_VF_ARRAY = 0,                                    ///< User V-F array
    ZES_VF_ARRAY_TYPE_DEFAULT_VF_ARRAY = 1,                                 ///< Default V-F array
    ZES_VF_ARRAY_TYPE_LIVE_VF_ARRAY = 2,                                    ///< Live V-F array
    ZES_VF_ARRAY_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_vf_array_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock properties
/// 
/// @details
///     - Information on the overclock domain type and all the contols that are
///       part of the domain.
typedef struct _zes_overclock_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_overclock_domain_t domainType;                                      ///< [out] The hardware block that this overclock domain controls (GPU,
                                                                            ///< VRAM, ...)
    uint32_t AvailableControls;                                             ///< [out] Returns the overclock controls that are supported (a bit for
                                                                            ///< each of enum ::zes_overclock_control_t). If no bits are set, the
                                                                            ///< domain doesn't support overclocking.
    zes_vf_program_type_t VFProgramType;                                    ///< [out] Type of V-F curve programming that is permitted:.
    uint32_t NumberOfVFPoints;                                              ///< [out] Number of VF points that can be programmed - max_num_points

} zes_overclock_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock Control properties
/// 
/// @details
///     - Provides all the control capabilities supported by the device for the
///       overclock domain.
typedef struct _zes_control_property_t
{
    double MinValue;                                                        ///< [out]  This provides information about the limits of the control value
                                                                            ///< so that the driver can calculate the set of valid values.
    double MaxValue;                                                        ///< [out]  This provides information about the limits of the control value
                                                                            ///< so that the driver can calculate the set of valid values.
    double StepValue;                                                       ///< [out]  This provides information about the limits of the control value
                                                                            ///< so that the driver can calculate the set of valid values.
    double RefValue;                                                        ///< [out] The reference value provides the anchor point, UIs can combine
                                                                            ///< this with the user offset request to show the anticipated improvement.
    double DefaultValue;                                                    ///< [out] The shipped out-of-box position of this control. Driver can
                                                                            ///< request this value at any time to return to the out-of-box behavior.

} zes_control_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclock VF properties
/// 
/// @details
///     - Provides all the VF capabilities supported by the device for the
///       overclock domain.
typedef struct _zes_vf_property_t
{
    double MinFreq;                                                         ///< [out] Read the minimum frequency that can be be programmed in the
                                                                            ///< custom V-F point..
    double MaxFreq;                                                         ///< [out] Read the maximum frequency that can be be programmed in the
                                                                            ///< custom V-F point..
    double StepFreq;                                                        ///< [out] Read the frequency step that can be be programmed in the custom
                                                                            ///< V-F point..
    double MinVolt;                                                         ///< [out] Read the minimum voltage that can be be programmed in the custom
                                                                            ///< V-F point..
    double MaxVolt;                                                         ///< [out] Read the maximum voltage that can be be programmed in the custom
                                                                            ///< V-F point..
    double StepVolt;                                                        ///< [out] Read the voltage step that can be be programmed in the custom
                                                                            ///< V-F point.

} zes_vf_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the overclock waiver.The overclock waiver setting is persistent
///        until the next pcode boot
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This product does not support overclocking
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceSetOverclockWaiver(
    zes_device_handle_t hDevice                                             ///< [in] Sysman handle of the device.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the list of supported overclock domains for this device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOverclockDomains`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetOverclockDomains(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pOverclockDomains                                             ///< [in,out] Returns the overclock domains that are supported (a bit for
                                                                            ///< each of enum ::zes_overclock_domain_t). If no bits are set, the device
                                                                            ///< doesn't support overclocking.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the list of supported overclock controls available for one of the
///        supported overclock domains on the device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OVERCLOCK_DOMAIN_ADM < domainType`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pAvailableControls`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetOverclockControls(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_overclock_domain_t domainType,                                      ///< [in] Domain type.
    uint32_t* pAvailableControls                                            ///< [in,out] Returns the overclock controls that are supported for the
                                                                            ///< specified overclock domain (a bit for each of enum
                                                                            ///< ::zes_overclock_control_t).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reset all overclock settings to default values (shipped = 1 or
///        manufacturing =0)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceResetOverclockSettings(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    ze_bool_t onShippedState                                                ///< [in] True will reset to shipped state; false will reset to
                                                                            ///< manufacturing state
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Determine the state of overclocking
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOverclockMode`
///         + `nullptr == pWaiverSetting`
///         + `nullptr == pOverclockState`
///         + `nullptr == pPendingAction`
///         + `nullptr == pPendingReset`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceReadOverclockState(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_overclock_mode_t* pOverclockMode,                                   ///< [out] One of overclock mode.
    ze_bool_t* pWaiverSetting,                                              ///< [out] Waiver setting: 0 = Waiver not set, 1 = waiver has been set.
    ze_bool_t* pOverclockState,                                             ///< [out] Current settings 0 =manufacturing state, 1= shipped state)..
    zes_pending_action_t* pPendingAction,                                   ///< [out] This enum is returned when the driver attempts to set an
                                                                            ///< overclock control or reset overclock settings.
    ze_bool_t* pPendingReset                                                ///< [out] Pending reset 0 =manufacturing state, 1= shipped state)..
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of overclock domains
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumOverclockDomains(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_overclock_handle_t* phDomainHandle                                  ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get overclock domain control properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pDomainProperties`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetDomainProperties(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_overclock_properties_t* pDomainProperties                           ///< [in,out] The overclock properties for the specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read overclock VF min,max and step values
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pVFProperties`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetDomainVFProperties(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_vf_property_t* pVFProperties                                        ///< [in,out] The VF min,max,step for a specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read overclock control values - min/max/step/default/ref
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OVERCLOCK_CONTROL_ACM_DISABLE < DomainControl`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pControlProperties`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetDomainControlProperties(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_overclock_control_t DomainControl,                                  ///< [in] Handle for the component.
    zes_control_property_t* pControlProperties                              ///< [in,out] overclock control values.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read the current value for a given overclock control
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OVERCLOCK_CONTROL_ACM_DISABLE < DomainControl`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pValue`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetControlCurrentValue(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component.
    zes_overclock_control_t DomainControl,                                  ///< [in] Overclock Control.
    double* pValue                                                          ///< [in,out] Getting overclock control value for the specified control.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read the the reset pending value for a given overclock control
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OVERCLOCK_CONTROL_ACM_DISABLE < DomainControl`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pValue`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetControlPendingValue(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_overclock_control_t DomainControl,                                  ///< [in] Overclock Control.
    double* pValue                                                          ///< [out] Returns the pending value for a given control. The units and
                                                                            ///< format of the value depend on the control type.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the value for a given overclock control
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OVERCLOCK_CONTROL_ACM_DISABLE < DomainControl`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pPendingAction`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockSetControlUserValue(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_overclock_control_t DomainControl,                                  ///< [in] Domain Control.
    double pValue,                                                          ///< [in] The new value of the control. The units and format of the value
                                                                            ///< depend on the control type.
    zes_pending_action_t* pPendingAction                                    ///< [out] Pending overclock setting.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Determine the state of an overclock control
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OVERCLOCK_CONTROL_ACM_DISABLE < DomainControl`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pControlState`
///         + `nullptr == pPendingAction`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetControlState(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_overclock_control_t DomainControl,                                  ///< [in] Domain Control.
    zes_control_state_t* pControlState,                                     ///< [out] Current overclock control state.
    zes_pending_action_t* pPendingAction                                    ///< [out] Pending overclock setting.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read the frequency or voltage of a V-F point from the default or
///        custom V-F curve.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_VF_TYPE_FREQ < VFType`
///         + `::ZES_VF_ARRAY_TYPE_LIVE_VF_ARRAY < VFArrayType`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == PointValue`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockGetVFPointValues(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_vf_type_t VFType,                                                   ///< [in] Voltage or Freqency point to read.
    zes_vf_array_type_t VFArrayType,                                        ///< [in] User,Default or Live VF array to read from
    uint32_t PointIndex,                                                    ///< [in] Point index - number between (0, max_num_points - 1).
    uint32_t* PointValue                                                    ///< [out] Returns the frequency in 1kHz units or voltage in millivolt
                                                                            ///< units from the custom V-F curve at the specified zero-based index 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Write the frequency or voltage of a V-F point to custom V-F curve.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDomainHandle`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_VF_TYPE_FREQ < VFType`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this control domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesOverclockSetVFPointValues(
    zes_overclock_handle_t hDomainHandle,                                   ///< [in] Handle for the component domain.
    zes_vf_type_t VFType,                                                   ///< [in] Voltage or Freqency point to read.
    uint32_t PointIndex,                                                    ///< [in] Point index - number between (0, max_num_points - 1).
    uint32_t PointValue                                                     ///< [in] Writes frequency in 1kHz units or voltage in millivolt units to
                                                                            ///< custom V-F curve at the specified zero-based index 
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region diagnostics
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Diagnostic results
typedef enum _zes_diag_result_t
{
    ZES_DIAG_RESULT_NO_ERRORS = 0,                                          ///< Diagnostic completed without finding errors to repair
    ZES_DIAG_RESULT_ABORT = 1,                                              ///< Diagnostic had problems running tests
    ZES_DIAG_RESULT_FAIL_CANT_REPAIR = 2,                                   ///< Diagnostic had problems setting up repairs
    ZES_DIAG_RESULT_REBOOT_FOR_REPAIR = 3,                                  ///< Diagnostics found errors, setup for repair and reboot is required to
                                                                            ///< complete the process
    ZES_DIAG_RESULT_FORCE_UINT32 = 0x7fffffff

} zes_diag_result_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_DIAG_FIRST_TEST_INDEX
/// @brief Diagnostic test index to use for the very first test.
#define ZES_DIAG_FIRST_TEST_INDEX  0x0
#endif // ZES_DIAG_FIRST_TEST_INDEX

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_DIAG_LAST_TEST_INDEX
/// @brief Diagnostic test index to use for the very last test.
#define ZES_DIAG_LAST_TEST_INDEX  0xFFFFFFFF
#endif // ZES_DIAG_LAST_TEST_INDEX

///////////////////////////////////////////////////////////////////////////////
/// @brief Diagnostic test
typedef struct _zes_diag_test_t
{
    uint32_t index;                                                         ///< [out] Index of the test
    char name[ZES_STRING_PROPERTY_SIZE];                                    ///< [out] Name of the test

} zes_diag_test_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Diagnostics test suite properties
typedef struct _zes_diag_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    char name[ZES_STRING_PROPERTY_SIZE];                                    ///< [out] Name of the diagnostics test suite
    ze_bool_t haveTests;                                                    ///< [out] Indicates if this test suite has individual tests which can be
                                                                            ///< run separately (use the function ::zesDiagnosticsGetTests() to get the
                                                                            ///< list of these tests)

} zes_diag_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of diagnostics test suites
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumDiagnosticTestSuites(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_diag_handle_t* phDiagnostics                                        ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties of a diagnostics test suite
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDiagnostics`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsGetProperties(
    zes_diag_handle_t hDiagnostics,                                         ///< [in] Handle for the component.
    zes_diag_properties_t* pProperties                                      ///< [in,out] Structure describing the properties of a diagnostics test
                                                                            ///< suite
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get individual tests that can be run separately. Not all test suites
///        permit running individual tests, check the `haveTests` member of
///        ::zes_diag_properties_t.
/// 
/// @details
///     - The list of available tests is returned in order of increasing test
///       index (see the `index` member of ::zes_diag_test_t).
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDiagnostics`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsGetTests(
    zes_diag_handle_t hDiagnostics,                                         ///< [in] Handle for the component.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of tests.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of tests that are available.
                                                                            ///< if count is greater than the number of tests that are available, then
                                                                            ///< the driver shall update the value with the correct number of tests.
    zes_diag_test_t* pTests                                                 ///< [in,out][optional][range(0, *pCount)] array of information about
                                                                            ///< individual tests sorted by increasing value of the `index` member of ::zes_diag_test_t.
                                                                            ///< if count is less than the number of tests that are available, then the
                                                                            ///< driver shall only retrieve that number of tests.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Run a diagnostics test suite, either all tests or a subset of tests.
/// 
/// @details
///     - WARNING: Running diagnostics may destroy current device state
///       information. Gracefully close any running workloads before initiating.
///     - To run all tests in a test suite, set start =
///       ::ZES_DIAG_FIRST_TEST_INDEX and end = ::ZES_DIAG_LAST_TEST_INDEX.
///     - If the test suite permits running individual tests, the `haveTests`
///       member of ::zes_diag_properties_t will be true. In this case, the
///       function ::zesDiagnosticsGetTests() can be called to get the list of
///       tests and corresponding indices that can be supplied to the arguments
///       start and end in this function.
///     - This function will block until the diagnostics have completed and
///       force reset based on result
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDiagnostics`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pResult`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform diagnostics.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsRunTests(
    zes_diag_handle_t hDiagnostics,                                         ///< [in] Handle for the component.
    uint32_t startIndex,                                                    ///< [in] The index of the first test to run. Set to
                                                                            ///< ::ZES_DIAG_FIRST_TEST_INDEX to start from the beginning.
    uint32_t endIndex,                                                      ///< [in] The index of the last test to run. Set to
                                                                            ///< ::ZES_DIAG_LAST_TEST_INDEX to complete all tests after the start test.
    zes_diag_result_t* pResult                                              ///< [in,out] The result of the diagnostics
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - ECC management
#if !defined(__GNUC__)
#pragma region ecc
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief ECC State
typedef enum _zes_device_ecc_state_t
{
    ZES_DEVICE_ECC_STATE_UNAVAILABLE = 0,                                   ///< None
    ZES_DEVICE_ECC_STATE_ENABLED = 1,                                       ///< ECC enabled.
    ZES_DEVICE_ECC_STATE_DISABLED = 2,                                      ///< ECC disabled.
    ZES_DEVICE_ECC_STATE_FORCE_UINT32 = 0x7fffffff

} zes_device_ecc_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief State Change Requirements
typedef enum _zes_device_action_t
{
    ZES_DEVICE_ACTION_NONE = 0,                                             ///< No action.
    ZES_DEVICE_ACTION_WARM_CARD_RESET = 1,                                  ///< Warm reset of the card.
    ZES_DEVICE_ACTION_COLD_CARD_RESET = 2,                                  ///< Cold reset of the card.
    ZES_DEVICE_ACTION_COLD_SYSTEM_REBOOT = 3,                               ///< Cold reboot of the system.
    ZES_DEVICE_ACTION_FORCE_UINT32 = 0x7fffffff

} zes_device_action_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief ECC State Descriptor
typedef struct _zes_device_ecc_desc_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_device_ecc_state_t state;                                           ///< [out] ECC state

} zes_device_ecc_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief ECC State
typedef struct _zes_device_ecc_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_device_ecc_state_t currentState;                                    ///< [out] Current ECC state
    zes_device_ecc_state_t pendingState;                                    ///< [out] Pending ECC state
    zes_device_action_t pendingAction;                                      ///< [out] Pending action

} zes_device_ecc_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Is ECC functionality available - true or false?
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pAvailable`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEccAvailable(
    zes_device_handle_t hDevice,                                            ///< [in] Handle for the component.
    ze_bool_t* pAvailable                                                   ///< [out] ECC functionality is available (true)/unavailable (false).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Is ECC support configurable - true or false?
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfigurable`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEccConfigurable(
    zes_device_handle_t hDevice,                                            ///< [in] Handle for the component.
    ze_bool_t* pConfigurable                                                ///< [out] ECC can be enabled/disabled (true)/enabled/disabled (false).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current ECC state, pending state, and pending action
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetEccState(
    zes_device_handle_t hDevice,                                            ///< [in] Handle for the component.
    zes_device_ecc_properties_t* pState                                     ///< [out] ECC state, pending state, and pending action for state change.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set new ECC state
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - ::zesDeviceGetState should be called to determine pending action
///       required to implement state change.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == newState`
///         + `nullptr == pState`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_DEVICE_ECC_STATE_DISABLED < newState->state`
///     - ::ZE_RESULT_WARNING_ACTION_REQUIRED
///         + User must look at the pendingAction attribute of pState & perform the action required to complete the ECC state change.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceSetEccState(
    zes_device_handle_t hDevice,                                            ///< [in] Handle for the component.
    const zes_device_ecc_desc_t* newState,                                  ///< [in] Pointer to desired ECC state.
    zes_device_ecc_properties_t* pState                                     ///< [out] ECC state, pending state, and pending action for state change.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Engine groups
#if !defined(__GNUC__)
#pragma region engine
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Accelerator engine groups
typedef enum _zes_engine_group_t
{
    ZES_ENGINE_GROUP_ALL = 0,                                               ///< Access information about all engines combined.
    ZES_ENGINE_GROUP_COMPUTE_ALL = 1,                                       ///< Access information about all compute engines combined. Compute engines
                                                                            ///< can only process compute kernels (no 3D content).
    ZES_ENGINE_GROUP_MEDIA_ALL = 2,                                         ///< Access information about all media engines combined.
    ZES_ENGINE_GROUP_COPY_ALL = 3,                                          ///< Access information about all copy (blitter) engines combined.
    ZES_ENGINE_GROUP_COMPUTE_SINGLE = 4,                                    ///< Access information about a single compute engine - this is an engine
                                                                            ///< that can process compute kernels. Note that single engines may share
                                                                            ///< the same underlying accelerator resources as other engines so activity
                                                                            ///< of such an engine may not be indicative of the underlying resource
                                                                            ///< utilization - use ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    ZES_ENGINE_GROUP_RENDER_SINGLE = 5,                                     ///< Access information about a single render engine - this is an engine
                                                                            ///< that can process both 3D content and compute kernels. Note that single
                                                                            ///< engines may share the same underlying accelerator resources as other
                                                                            ///< engines so activity of such an engine may not be indicative of the
                                                                            ///< underlying resource utilization - use
                                                                            ///< ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    ZES_ENGINE_GROUP_MEDIA_DECODE_SINGLE = 6,                               ///< [DEPRECATED] No longer supported.
    ZES_ENGINE_GROUP_MEDIA_ENCODE_SINGLE = 7,                               ///< [DEPRECATED] No longer supported.
    ZES_ENGINE_GROUP_COPY_SINGLE = 8,                                       ///< Access information about a single media encode engine. Note that
                                                                            ///< single engines may share the same underlying accelerator resources as
                                                                            ///< other engines so activity of such an engine may not be indicative of
                                                                            ///< the underlying resource utilization - use ::ZES_ENGINE_GROUP_COPY_ALL
                                                                            ///< for that.
    ZES_ENGINE_GROUP_MEDIA_ENHANCEMENT_SINGLE = 9,                          ///< Access information about a single media enhancement engine. Note that
                                                                            ///< single engines may share the same underlying accelerator resources as
                                                                            ///< other engines so activity of such an engine may not be indicative of
                                                                            ///< the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                                            ///< for that.
    ZES_ENGINE_GROUP_3D_SINGLE = 10,                                        ///< [DEPRECATED] No longer supported.
    ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL = 11,                            ///< [DEPRECATED] No longer supported.
    ZES_ENGINE_GROUP_RENDER_ALL = 12,                                       ///< Access information about all render engines combined. Render engines
                                                                            ///< are those than process both 3D content and compute kernels.
    ZES_ENGINE_GROUP_3D_ALL = 13,                                           ///< [DEPRECATED] No longer supported.
    ZES_ENGINE_GROUP_MEDIA_CODEC_SINGLE = 14,                               ///< Access information about a single media engine. Note that single
                                                                            ///< engines may share the same underlying accelerator resources as other
                                                                            ///< engines so activity of such an engine may not be indicative of the
                                                                            ///< underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL for
                                                                            ///< that.
    ZES_ENGINE_GROUP_FORCE_UINT32 = 0x7fffffff

} zes_engine_group_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Engine group properties
typedef struct _zes_engine_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_engine_group_t type;                                                ///< [out] The engine group
    ze_bool_t onSubdevice;                                                  ///< [out] True if this resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device

} zes_engine_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Engine activity counters
/// 
/// @details
///     - Percent utilization is calculated by taking two snapshots (s1, s2) and
///       using the equation: %util = (s2.activeTime - s1.activeTime) /
///       (s2.timestamp - s1.timestamp)
typedef struct _zes_engine_stats_t
{
    uint64_t activeTime;                                                    ///< [out] Monotonic counter where the resource is actively running workloads.
                                                                            ///< Time units are implementation specific since the activeTime value is
                                                                            ///< only intended for calculating utilization percentage as noted above.
    uint64_t timestamp;                                                     ///< [out] Monotonic counter when activeTime counter was sampled.
                                                                            ///< This timestamp should only be used to calculate delta between
                                                                            ///< snapshots of this structure.
                                                                            ///< Never take the delta of this timestamp with the timestamp from a
                                                                            ///< different structure since they are not guaranteed to have the same base.
                                                                            ///< The absolute value of the timestamp is only valid during within the
                                                                            ///< application and may be different on the next execution.

} zes_engine_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of engine groups
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumEngineGroups(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_engine_handle_t* phEngine                                           ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get engine group properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEngine`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetProperties(
    zes_engine_handle_t hEngine,                                            ///< [in] Handle for the component.
    zes_engine_properties_t* pProperties                                    ///< [in,out] The properties for the specified engine group.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the activity stats for an engine group.
/// 
/// @details
///     - This function also returns the engine activity inside a Virtual
///       Machine (VM), in the presence of hardware virtualization (SRIOV)
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEngine`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pStats`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetActivity(
    zes_engine_handle_t hEngine,                                            ///< [in] Handle for the component.
    zes_engine_stats_t* pStats                                              ///< [in,out] Will contain a snapshot of the engine group activity
                                                                            ///< counters.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Event management
#if !defined(__GNUC__)
#pragma region events
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Event types
typedef uint32_t zes_event_type_flags_t;
typedef enum _zes_event_type_flag_t
{
    ZES_EVENT_TYPE_FLAG_DEVICE_DETACH = ZE_BIT(0),                          ///< Event is triggered when the device is no longer available (due to a
                                                                            ///< reset or being disabled).
    ZES_EVENT_TYPE_FLAG_DEVICE_ATTACH = ZE_BIT(1),                          ///< Event is triggered after the device is available again.
    ZES_EVENT_TYPE_FLAG_DEVICE_SLEEP_STATE_ENTER = ZE_BIT(2),               ///< Event is triggered when the driver is about to put the device into a
                                                                            ///< deep sleep state
    ZES_EVENT_TYPE_FLAG_DEVICE_SLEEP_STATE_EXIT = ZE_BIT(3),                ///< Event is triggered when the driver is waking the device up from a deep
                                                                            ///< sleep state
    ZES_EVENT_TYPE_FLAG_FREQ_THROTTLED = ZE_BIT(4),                         ///< Event is triggered when the frequency starts being throttled
    ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED = ZE_BIT(5),               ///< Event is triggered when the energy consumption threshold is reached
                                                                            ///< (use ::zesPowerSetEnergyThreshold() to configure).
    ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL = ZE_BIT(6),                          ///< Event is triggered when the critical temperature is reached (use
                                                                            ///< ::zesTemperatureSetConfig() to configure - disabled by default).
    ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 = ZE_BIT(7),                        ///< Event is triggered when the temperature crosses threshold 1 (use
                                                                            ///< ::zesTemperatureSetConfig() to configure - disabled by default).
    ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 = ZE_BIT(8),                        ///< Event is triggered when the temperature crosses threshold 2 (use
                                                                            ///< ::zesTemperatureSetConfig() to configure - disabled by default).
    ZES_EVENT_TYPE_FLAG_MEM_HEALTH = ZE_BIT(9),                             ///< Event is triggered when the health of device memory changes.
    ZES_EVENT_TYPE_FLAG_FABRIC_PORT_HEALTH = ZE_BIT(10),                    ///< Event is triggered when the health of fabric ports change.
    ZES_EVENT_TYPE_FLAG_PCI_LINK_HEALTH = ZE_BIT(11),                       ///< Event is triggered when the health of the PCI link changes.
    ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS = ZE_BIT(12),                ///< Event is triggered when accelerator RAS correctable errors cross
                                                                            ///< thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                                            ///< default).
    ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS = ZE_BIT(13),              ///< Event is triggered when accelerator RAS uncorrectable errors cross
                                                                            ///< thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                                            ///< default).
    ZES_EVENT_TYPE_FLAG_DEVICE_RESET_REQUIRED = ZE_BIT(14),                 ///< Event is triggered when the device needs to be reset (use
                                                                            ///< ::zesDeviceGetState() to determine the reasons for the reset).
    ZES_EVENT_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_event_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Specify the list of events to listen to for a given device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7fff < events`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEventRegister(
    zes_device_handle_t hDevice,                                            ///< [in] The device handle.
    zes_event_type_flags_t events                                           ///< [in] List of events to listen to.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for events to be received from a one or more devices.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevices`
///         + `nullptr == pNumDeviceEvents`
///         + `nullptr == pEvents`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to listen to events.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + One or more of the supplied device handles belongs to a different driver.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverEventListen(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    uint32_t timeout,                                                       ///< [in] if non-zero, then indicates the maximum time (in milliseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then will check status and return immediately;
                                                                            ///< if `UINT32_MAX`, then function will not return until events arrive.
    uint32_t count,                                                         ///< [in] Number of device handles in phDevices.
    zes_device_handle_t* phDevices,                                         ///< [in][range(0, count)] Device handles to listen to for events. Only
                                                                            ///< devices from the provided driver handle can be specified in this list.
    uint32_t* pNumDeviceEvents,                                             ///< [in,out] Will contain the actual number of devices in phDevices that
                                                                            ///< generated events. If non-zero, check pEvents to determine the devices
                                                                            ///< and events that were received.
    zes_event_type_flags_t* pEvents                                         ///< [in,out] An array that will continue the list of events for each
                                                                            ///< device listened in phDevices.
                                                                            ///< This array must be at least as big as count.
                                                                            ///< For every device handle in phDevices, this will provide the events
                                                                            ///< that occurred for that device at the same position in this array. If
                                                                            ///< no event was received for a given device, the corresponding array
                                                                            ///< entry will be zero.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for events to be received from a one or more devices.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevices`
///         + `nullptr == pNumDeviceEvents`
///         + `nullptr == pEvents`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to listen to events.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + One or more of the supplied device handles belongs to a different driver.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverEventListenEx(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    uint64_t timeout,                                                       ///< [in] if non-zero, then indicates the maximum time (in milliseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then will check status and return immediately;
                                                                            ///< if `UINT64_MAX`, then function will not return until events arrive.
    uint32_t count,                                                         ///< [in] Number of device handles in phDevices.
    zes_device_handle_t* phDevices,                                         ///< [in][range(0, count)] Device handles to listen to for events. Only
                                                                            ///< devices from the provided driver handle can be specified in this list.
    uint32_t* pNumDeviceEvents,                                             ///< [in,out] Will contain the actual number of devices in phDevices that
                                                                            ///< generated events. If non-zero, check pEvents to determine the devices
                                                                            ///< and events that were received.
    zes_event_type_flags_t* pEvents                                         ///< [in,out] An array that will continue the list of events for each
                                                                            ///< device listened in phDevices.
                                                                            ///< This array must be at least as big as count.
                                                                            ///< For every device handle in phDevices, this will provide the events
                                                                            ///< that occurred for that device at the same position in this array. If
                                                                            ///< no event was received for a given device, the corresponding array
                                                                            ///< entry will be zero.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region fabric
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_FABRIC_PORT_MODEL_SIZE
/// @brief Maximum Fabric port model string size
#define ZES_MAX_FABRIC_PORT_MODEL_SIZE  256
#endif // ZES_MAX_FABRIC_PORT_MODEL_SIZE

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_FABRIC_LINK_TYPE_SIZE
/// @brief Maximum size of the buffer that will return information about link
///        types
#define ZES_MAX_FABRIC_LINK_TYPE_SIZE  256
#endif // ZES_MAX_FABRIC_LINK_TYPE_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port status
typedef enum _zes_fabric_port_status_t
{
    ZES_FABRIC_PORT_STATUS_UNKNOWN = 0,                                     ///< The port status cannot be determined
    ZES_FABRIC_PORT_STATUS_HEALTHY = 1,                                     ///< The port is up and operating as expected
    ZES_FABRIC_PORT_STATUS_DEGRADED = 2,                                    ///< The port is up but has quality and/or speed degradation
    ZES_FABRIC_PORT_STATUS_FAILED = 3,                                      ///< Port connection instabilities are preventing workloads making forward
                                                                            ///< progress
    ZES_FABRIC_PORT_STATUS_DISABLED = 4,                                    ///< The port is configured down
    ZES_FABRIC_PORT_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port quality degradation reasons
typedef uint32_t zes_fabric_port_qual_issue_flags_t;
typedef enum _zes_fabric_port_qual_issue_flag_t
{
    ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_LINK_ERRORS = ZE_BIT(0),                ///< Excessive link errors are occurring
    ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_SPEED = ZE_BIT(1),                      ///< There is a degradation in the bitrate and/or width of the link
    ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_qual_issue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port failure reasons
typedef uint32_t zes_fabric_port_failure_flags_t;
typedef enum _zes_fabric_port_failure_flag_t
{
    ZES_FABRIC_PORT_FAILURE_FLAG_FAILED = ZE_BIT(0),                        ///< A previously operating link has failed. Hardware will automatically
                                                                            ///< retrain this port. This state will persist until either the physical
                                                                            ///< connection is removed or the link trains successfully.
    ZES_FABRIC_PORT_FAILURE_FLAG_TRAINING_TIMEOUT = ZE_BIT(1),              ///< A connection has not been established within an expected time.
                                                                            ///< Hardware will continue to attempt port training. This status will
                                                                            ///< persist until either the physical connection is removed or the link
                                                                            ///< successfully trains.
    ZES_FABRIC_PORT_FAILURE_FLAG_FLAPPING = ZE_BIT(2),                      ///< Port has excessively trained and then transitioned down for some
                                                                            ///< period of time. Driver will allow port to continue to train, but will
                                                                            ///< not enable the port for use until the port has been disabled and
                                                                            ///< subsequently re-enabled using ::zesFabricPortSetConfig().
    ZES_FABRIC_PORT_FAILURE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_failure_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Unique identifier for a fabric port
/// 
/// @details
///     - This not a universal identifier. The identified is garanteed to be
///       unique for the current hardware configuration of the system. Changes
///       in the hardware may result in a different identifier for a given port.
///     - The main purpose of this identifier to build up an instantaneous
///       topology map of system connectivity. An application should enumerate
///       all fabric ports and match the `remotePortId` member of
///       ::zes_fabric_port_state_t to the `portId` member of
///       ::zes_fabric_port_properties_t.
typedef struct _zes_fabric_port_id_t
{
    uint32_t fabricId;                                                      ///< [out] Unique identifier for the fabric end-point
    uint32_t attachId;                                                      ///< [out] Unique identifier for the device attachment point
    uint8_t portNumber;                                                     ///< [out] The logical port number (this is typically marked somewhere on
                                                                            ///< the physical device)

} zes_fabric_port_id_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port speed in one direction
typedef struct _zes_fabric_port_speed_t
{
    int64_t bitRate;                                                        ///< [out] Bits/sec that the link is operating at. A value of -1 means that
                                                                            ///< this property is unknown.
    int32_t width;                                                          ///< [out] The number of lanes. A value of -1 means that this property is
                                                                            ///< unknown.

} zes_fabric_port_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port properties
typedef struct _zes_fabric_port_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    char model[ZES_MAX_FABRIC_PORT_MODEL_SIZE];                             ///< [out] Description of port technology. Will be set to the string
                                                                            ///< "unkown" if this cannot be determined for this port.
    ze_bool_t onSubdevice;                                                  ///< [out] True if the port is located on a sub-device; false means that
                                                                            ///< the port is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    zes_fabric_port_id_t portId;                                            ///< [out] The unique port identifier
    zes_fabric_port_speed_t maxRxSpeed;                                     ///< [out] Maximum speed supported by the receive side of the port (sum of
                                                                            ///< all lanes)
    zes_fabric_port_speed_t maxTxSpeed;                                     ///< [out] Maximum speed supported by the transmit side of the port (sum of
                                                                            ///< all lanes)

} zes_fabric_port_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Provides information about the fabric link attached to a port
typedef struct _zes_fabric_link_type_t
{
    char desc[ZES_MAX_FABRIC_LINK_TYPE_SIZE];                               ///< [out] Description of link technology. Will be set to the string
                                                                            ///< "unkown" if this cannot be determined for this link.

} zes_fabric_link_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port configuration
typedef struct _zes_fabric_port_config_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t enabled;                                                      ///< [in,out] Port is configured up/down
    ze_bool_t beaconing;                                                    ///< [in,out] Beaconing is configured on/off

} zes_fabric_port_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port state
typedef struct _zes_fabric_port_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_fabric_port_status_t status;                                        ///< [out] The current status of the port
    zes_fabric_port_qual_issue_flags_t qualityIssues;                       ///< [out] If status is ::ZES_FABRIC_PORT_STATUS_DEGRADED,
                                                                            ///< then this gives a combination of ::zes_fabric_port_qual_issue_flag_t
                                                                            ///< for quality issues that have been detected;
                                                                            ///< otherwise, 0 indicates there are no quality issues with the link at
                                                                            ///< this time.
    zes_fabric_port_failure_flags_t failureReasons;                         ///< [out] If status is ::ZES_FABRIC_PORT_STATUS_FAILED,
                                                                            ///< then this gives a combination of ::zes_fabric_port_failure_flag_t for
                                                                            ///< reasons for the connection instability;
                                                                            ///< otherwise, 0 indicates there are no connection stability issues at
                                                                            ///< this time.
    zes_fabric_port_id_t remotePortId;                                      ///< [out] The unique port identifier for the remote connection point if
                                                                            ///< status is ::ZES_FABRIC_PORT_STATUS_HEALTHY,
                                                                            ///< ::ZES_FABRIC_PORT_STATUS_DEGRADED or ::ZES_FABRIC_PORT_STATUS_FAILED
    zes_fabric_port_speed_t rxSpeed;                                        ///< [out] Current maximum receive speed (sum of all lanes)
    zes_fabric_port_speed_t txSpeed;                                        ///< [out] Current maximum transmit speed (sum of all lanes)

} zes_fabric_port_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port throughput.
typedef struct _zes_fabric_port_throughput_t
{
    uint64_t timestamp;                                                     ///< [out] Monotonic timestamp counter in microseconds when the measurement
                                                                            ///< was made.
                                                                            ///< This timestamp should only be used to calculate delta time between
                                                                            ///< snapshots of this structure.
                                                                            ///< Never take the delta of this timestamp with the timestamp from a
                                                                            ///< different structure since they are not guaranteed to have the same base.
                                                                            ///< The absolute value of the timestamp is only valid during within the
                                                                            ///< application and may be different on the next execution.
    uint64_t rxCounter;                                                     ///< [out] Monotonic counter for the number of bytes received (sum of all
                                                                            ///< lanes). This includes all protocol overhead, not only the GPU traffic.
    uint64_t txCounter;                                                     ///< [out] Monotonic counter for the number of bytes transmitted (sum of
                                                                            ///< all lanes). This includes all protocol overhead, not only the GPU
                                                                            ///< traffic.

} zes_fabric_port_throughput_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric Port Error Counters
typedef struct _zes_fabric_port_error_counters_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t linkFailureCount;                                              ///< [out] Link Failure Error Count reported per port
    uint64_t fwCommErrorCount;                                              ///< [out] Firmware Communication Error Count reported per device
    uint64_t fwErrorCount;                                                  ///< [out] Firmware reported Error Count reported per device
    uint64_t linkDegradeCount;                                              ///< [out] Link Degrade Error Count reported per port

} zes_fabric_port_error_counters_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of Fabric ports in a device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFabricPorts(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_fabric_port_handle_t* phPort                                        ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetProperties(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    zes_fabric_port_properties_t* pProperties                               ///< [in,out] Will contain properties of the Fabric Port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port link type
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLinkType`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetLinkType(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    zes_fabric_link_type_t* pLinkType                                       ///< [in,out] Will contain details about the link attached to the Fabric
                                                                            ///< port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port configuration
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetConfig(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    zes_fabric_port_config_t* pConfig                                       ///< [in,out] Will contain configuration of the Fabric Port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set Fabric port configuration
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortSetConfig(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    const zes_fabric_port_config_t* pConfig                                 ///< [in] Contains new configuration of the Fabric Port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port state - status (health/degraded/failed/disabled),
///        reasons for link degradation or instability, current rx/tx speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetState(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    zes_fabric_port_state_t* pState                                         ///< [in,out] Will contain the current state of the Fabric Port
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port throughput
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pThroughput`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetThroughput(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    zes_fabric_port_throughput_t* pThroughput                               ///< [in,out] Will contain the Fabric port throughput counters.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric Port Error Counters
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - The memory backing the arrays for phPorts and ppThroughputs must be
///       allocated in system memory by the user who is also responsible for
///       releasing them when they are no longer needed.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pErrors`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetFabricErrorCounters(
    zes_fabric_port_handle_t hPort,                                         ///< [in] Handle for the component.
    zes_fabric_port_error_counters_t* pErrors                               ///< [in,out] Will contain the Fabric port Error counters.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port throughput from multiple ports in a single call
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phPort`
///         + `nullptr == pThroughput`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetMultiPortThroughput(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t numPorts,                                                      ///< [in] Number of ports enumerated in function ::zesDeviceEnumFabricPorts
    zes_fabric_port_handle_t* phPort,                                       ///< [in][range(0, numPorts)] array of fabric port handles provided by user
                                                                            ///< to gather throughput values. 
    zes_fabric_port_throughput_t** pThroughput                              ///< [out][range(0, numPorts)] array of fabric port throughput counters
                                                                            ///< from multiple ports of type ::zes_fabric_port_throughput_t.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region fan
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Fan resource speed mode
typedef enum _zes_fan_speed_mode_t
{
    ZES_FAN_SPEED_MODE_DEFAULT = 0,                                         ///< The fan speed is operating using the hardware default settings
    ZES_FAN_SPEED_MODE_FIXED = 1,                                           ///< The fan speed is currently set to a fixed value
    ZES_FAN_SPEED_MODE_TABLE = 2,                                           ///< The fan speed is currently controlled dynamically by hardware based on
                                                                            ///< a temp/speed table
    ZES_FAN_SPEED_MODE_FORCE_UINT32 = 0x7fffffff

} zes_fan_speed_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan speed units
typedef enum _zes_fan_speed_units_t
{
    ZES_FAN_SPEED_UNITS_RPM = 0,                                            ///< The fan speed is in units of revolutions per minute (rpm)
    ZES_FAN_SPEED_UNITS_PERCENT = 1,                                        ///< The fan speed is a percentage of the maximum speed of the fan
    ZES_FAN_SPEED_UNITS_FORCE_UINT32 = 0x7fffffff

} zes_fan_speed_units_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan speed
typedef struct _zes_fan_speed_t
{
    int32_t speed;                                                          ///< [in,out] The speed of the fan. On output, a value of -1 indicates that
                                                                            ///< there is no fixed fan speed setting.
    zes_fan_speed_units_t units;                                            ///< [in,out] The units that the fan speed is expressed in. On output, if
                                                                            ///< fan speed is -1 then units should be ignored.

} zes_fan_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan temperature/speed pair
typedef struct _zes_fan_temp_speed_t
{
    uint32_t temperature;                                                   ///< [in,out] Temperature in degrees Celsius.
    zes_fan_speed_t speed;                                                  ///< [in,out] The speed of the fan

} zes_fan_temp_speed_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_FAN_TEMP_SPEED_PAIR_COUNT
/// @brief Maximum number of fan temperature/speed pairs in the fan speed table.
#define ZES_FAN_TEMP_SPEED_PAIR_COUNT  32
#endif // ZES_FAN_TEMP_SPEED_PAIR_COUNT

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan speed table
typedef struct _zes_fan_speed_table_t
{
    int32_t numPoints;                                                      ///< [in,out] The number of valid points in the fan speed table. 0 means
                                                                            ///< that there is no fan speed table configured. -1 means that a fan speed
                                                                            ///< table is not supported by the hardware.
    zes_fan_temp_speed_t table[ZES_FAN_TEMP_SPEED_PAIR_COUNT];              ///< [in,out] Array of temperature/fan speed pairs. The table is ordered
                                                                            ///< based on temperature from lowest to highest.

} zes_fan_speed_table_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan properties
typedef struct _zes_fan_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                                                   ///< [out] Indicates if software can control the fan speed assuming the
                                                                            ///< user has permissions
    uint32_t supportedModes;                                                ///< [out] Bitfield of supported fan configuration modes
                                                                            ///< (1<<::zes_fan_speed_mode_t)
    uint32_t supportedUnits;                                                ///< [out] Bitfield of supported fan speed units
                                                                            ///< (1<<::zes_fan_speed_units_t)
    int32_t maxRPM;                                                         ///< [out] The maximum RPM of the fan. A value of -1 means that this
                                                                            ///< property is unknown. 
    int32_t maxPoints;                                                      ///< [out] The maximum number of points in the fan temp/speed table. A
                                                                            ///< value of -1 means that this fan doesn't support providing a temp/speed
                                                                            ///< table.

} zes_fan_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan configuration
typedef struct _zes_fan_config_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_fan_speed_mode_t mode;                                              ///< [in,out] The fan speed mode (fixed, temp-speed table)
    zes_fan_speed_t speedFixed;                                             ///< [in,out] The current fixed fan speed setting
    zes_fan_speed_table_t speedTable;                                       ///< [out] A table containing temperature/speed pairs

} zes_fan_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of fans
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFans(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_fan_handle_t* phFan                                                 ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get fan properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetProperties(
    zes_fan_handle_t hFan,                                                  ///< [in] Handle for the component.
    zes_fan_properties_t* pProperties                                       ///< [in,out] Will contain the properties of the fan.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get fan configurations and the current fan speed mode (default, fixed,
///        temp-speed table)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetConfig(
    zes_fan_handle_t hFan,                                                  ///< [in] Handle for the component.
    zes_fan_config_t* pConfig                                               ///< [in,out] Will contain the current configuration of the fan.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Configure the fan to run with hardware factory settings (set mode to
///        ::ZES_FAN_SPEED_MODE_DEFAULT)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetDefaultMode(
    zes_fan_handle_t hFan                                                   ///< [in] Handle for the component.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Configure the fan to rotate at a fixed speed (set mode to
///        ::ZES_FAN_SPEED_MODE_FIXED)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == speed`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Fixing the fan speed not supported by the hardware or the fan speed units are not supported. See the `supportedModes` and `supportedUnits` members of ::zes_fan_properties_t.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetFixedSpeedMode(
    zes_fan_handle_t hFan,                                                  ///< [in] Handle for the component.
    const zes_fan_speed_t* speed                                            ///< [in] The fixed fan speed setting
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Configure the fan to adjust speed based on a temperature/speed table
///        (set mode to ::ZES_FAN_SPEED_MODE_TABLE)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == speedTable`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + The temperature/speed pairs in the array are not sorted on temperature from lowest to highest.
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Fan speed table not supported by the hardware or the fan speed units are not supported. See the `supportedModes` and `supportedUnits` members of ::zes_fan_properties_t.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetSpeedTableMode(
    zes_fan_handle_t hFan,                                                  ///< [in] Handle for the component.
    const zes_fan_speed_table_t* speedTable                                 ///< [in] A table containing temperature/speed pairs.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current state of a fan - current mode and speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_FAN_SPEED_UNITS_PERCENT < units`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSpeed`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + The requested fan speed units are not supported. See the `supportedUnits` member of ::zes_fan_properties_t.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetState(
    zes_fan_handle_t hFan,                                                  ///< [in] Handle for the component.
    zes_fan_speed_units_t units,                                            ///< [in] The units in which the fan speed should be returned.
    int32_t* pSpeed                                                         ///< [in,out] Will contain the current speed of the fan in the units
                                                                            ///< requested. A value of -1 indicates that the fan speed cannot be
                                                                            ///< measured.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region firmware
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Firmware properties
typedef struct _zes_firmware_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                                                   ///< [out] Indicates if software can flash the firmware assuming the user
                                                                            ///< has permissions
    char name[ZES_STRING_PROPERTY_SIZE];                                    ///< [out] NULL terminated string value. The string "unknown" will be
                                                                            ///< returned if this property cannot be determined.
    char version[ZES_STRING_PROPERTY_SIZE];                                 ///< [out] NULL terminated string value. The string "unknown" will be
                                                                            ///< returned if this property cannot be determined.

} zes_firmware_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of firmwares
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFirmwares(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_firmware_handle_t* phFirmware                                       ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get firmware properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFirmware`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareGetProperties(
    zes_firmware_handle_t hFirmware,                                        ///< [in] Handle for the component.
    zes_firmware_properties_t* pProperties                                  ///< [in,out] Pointer to an array that will hold the properties of the
                                                                            ///< firmware
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Flash a new firmware image
/// 
/// @details
///     - Any running workload must be gracefully closed before invoking this
///       function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - This is a non-blocking call. Application may call
///       ::zesFirmwareGetFlashProgress to get completion status.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFirmware`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pImage`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform this operation.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareFlash(
    zes_firmware_handle_t hFirmware,                                        ///< [in] Handle for the component.
    void* pImage,                                                           ///< [in] Image of the new firmware to flash.
    uint32_t size                                                           ///< [in] Size of the flash image.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Firmware Flash Progress
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFirmware`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCompletionPercent`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareGetFlashProgress(
    zes_firmware_handle_t hFirmware,                                        ///< [in] Handle for the component.
    uint32_t* pCompletionPercent                                            ///< [in,out] Pointer to the Completion Percentage of Firmware Update
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Frequency domains
#if !defined(__GNUC__)
#pragma region frequency
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency domains.
typedef enum _zes_freq_domain_t
{
    ZES_FREQ_DOMAIN_GPU = 0,                                                ///< GPU Core Domain.
    ZES_FREQ_DOMAIN_MEMORY = 1,                                             ///< Local Memory Domain.
    ZES_FREQ_DOMAIN_MEDIA = 2,                                              ///< GPU Media Domain.
    ZES_FREQ_DOMAIN_FORCE_UINT32 = 0x7fffffff

} zes_freq_domain_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency properties
/// 
/// @details
///     - Indicates if this frequency domain can be overclocked (if true,
///       functions such as ::zesFrequencyOcSetFrequencyTarget() are supported).
///     - The min/max hardware frequencies are specified for non-overclock
///       configurations. For overclock configurations, use
///       ::zesFrequencyOcGetFrequencyTarget() to determine the maximum
///       frequency that can be requested.
typedef struct _zes_freq_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_freq_domain_t type;                                                 ///< [out] The hardware block that this frequency domain controls (GPU,
                                                                            ///< memory, ...)
    ze_bool_t onSubdevice;                                                  ///< [out] True if this resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                                                   ///< [out] Indicates if software can control the frequency of this domain
                                                                            ///< assuming the user has permissions
    ze_bool_t isThrottleEventSupported;                                     ///< [out] Indicates if software can register to receive event
                                                                            ///< ::ZES_EVENT_TYPE_FLAG_FREQ_THROTTLED
    double min;                                                             ///< [out] The minimum hardware clock frequency in units of MHz.
    double max;                                                             ///< [out] The maximum non-overclock hardware clock frequency in units of
                                                                            ///< MHz.

} zes_freq_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency range between which the hardware can operate.
/// 
/// @details
///     - When setting limits, they will be clamped to the hardware limits.
///     - When setting limits, ensure that the max frequency is greater than or
///       equal to the min frequency specified.
///     - When setting limits to return to factory settings, specify -1 for both
///       the min and max limit.
typedef struct _zes_freq_range_t
{
    double min;                                                             ///< [in,out] The min frequency in MHz below which hardware frequency
                                                                            ///< management will not request frequencies. On input, setting to 0 will
                                                                            ///< permit the frequency to go down to the hardware minimum while setting
                                                                            ///< to -1 will return the min frequency limit to the factory value (can be
                                                                            ///< larger than the hardware min). On output, a negative value indicates
                                                                            ///< that no external minimum frequency limit is in effect.
    double max;                                                             ///< [in,out] The max frequency in MHz above which hardware frequency
                                                                            ///< management will not request frequencies. On input, setting to 0 or a
                                                                            ///< very big number will permit the frequency to go all the way up to the
                                                                            ///< hardware maximum while setting to -1 will return the max frequency to
                                                                            ///< the factory value (which can be less than the hardware max). On
                                                                            ///< output, a negative number indicates that no external maximum frequency
                                                                            ///< limit is in effect.

} zes_freq_range_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency throttle reasons
typedef uint32_t zes_freq_throttle_reason_flags_t;
typedef enum _zes_freq_throttle_reason_flag_t
{
    ZES_FREQ_THROTTLE_REASON_FLAG_AVE_PWR_CAP = ZE_BIT(0),                  ///< frequency throttled due to average power excursion (PL1)
    ZES_FREQ_THROTTLE_REASON_FLAG_BURST_PWR_CAP = ZE_BIT(1),                ///< frequency throttled due to burst power excursion (PL2)
    ZES_FREQ_THROTTLE_REASON_FLAG_CURRENT_LIMIT = ZE_BIT(2),                ///< frequency throttled due to current excursion (PL4)
    ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL_LIMIT = ZE_BIT(3),                ///< frequency throttled due to thermal excursion (T > TjMax)
    ZES_FREQ_THROTTLE_REASON_FLAG_PSU_ALERT = ZE_BIT(4),                    ///< frequency throttled due to power supply assertion
    ZES_FREQ_THROTTLE_REASON_FLAG_SW_RANGE = ZE_BIT(5),                     ///< frequency throttled due to software supplied frequency range
    ZES_FREQ_THROTTLE_REASON_FLAG_HW_RANGE = ZE_BIT(6),                     ///< frequency throttled due to a sub block that has a lower frequency
                                                                            ///< range when it receives clocks
    ZES_FREQ_THROTTLE_REASON_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_freq_throttle_reason_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency state
typedef struct _zes_freq_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    double currentVoltage;                                                  ///< [out] Current voltage in Volts. A negative value indicates that this
                                                                            ///< property is not known.
    double request;                                                         ///< [out] The current frequency request in MHz. A negative value indicates
                                                                            ///< that this property is not known.
    double tdp;                                                             ///< [out] The maximum frequency in MHz supported under the current TDP
                                                                            ///< conditions. This fluctuates dynamically based on the power and thermal
                                                                            ///< limits of the part. A negative value indicates that this property is
                                                                            ///< not known.
    double efficient;                                                       ///< [out] The efficient minimum frequency in MHz. A negative value
                                                                            ///< indicates that this property is not known.
    double actual;                                                          ///< [out] The resolved frequency in MHz. A negative value indicates that
                                                                            ///< this property is not known.
    zes_freq_throttle_reason_flags_t throttleReasons;                       ///< [out] The reasons that the frequency is being limited by the hardware.
                                                                            ///< Returns 0 (frequency not throttled) or a combination of ::zes_freq_throttle_reason_flag_t.

} zes_freq_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency throttle time snapshot
/// 
/// @details
///     - Percent time throttled is calculated by taking two snapshots (s1, s2)
///       and using the equation: %throttled = (s2.throttleTime -
///       s1.throttleTime) / (s2.timestamp - s1.timestamp)
typedef struct _zes_freq_throttle_time_t
{
    uint64_t throttleTime;                                                  ///< [out] The monotonic counter of time in microseconds that the frequency
                                                                            ///< has been limited by the hardware.
    uint64_t timestamp;                                                     ///< [out] Microsecond timestamp when throttleTime was captured.
                                                                            ///< This timestamp should only be used to calculate delta time between
                                                                            ///< snapshots of this structure.
                                                                            ///< Never take the delta of this timestamp with the timestamp from a
                                                                            ///< different structure since they are not guaranteed to have the same base.
                                                                            ///< The absolute value of the timestamp is only valid during within the
                                                                            ///< application and may be different on the next execution.

} zes_freq_throttle_time_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclocking modes
/// 
/// @details
///     - [DEPRECATED] No longer supported.
typedef enum _zes_oc_mode_t
{
    ZES_OC_MODE_OFF = 0,                                                    ///< Overclocking if off - hardware is running using factory default
                                                                            ///< voltages/frequencies.
    ZES_OC_MODE_OVERRIDE = 1,                                               ///< Overclock override mode - In this mode, a fixed user-supplied voltage
                                                                            ///< is applied independent of the frequency request. The maximum permitted
                                                                            ///< frequency can also be increased. This mode disables INTERPOLATIVE and
                                                                            ///< FIXED modes.
    ZES_OC_MODE_INTERPOLATIVE = 2,                                          ///< Overclock interpolative mode - In this mode, the voltage/frequency
                                                                            ///< curve can be extended with a new voltage/frequency point that will be
                                                                            ///< interpolated. The existing voltage/frequency points can also be offset
                                                                            ///< (up or down) by a fixed voltage. This mode disables FIXED and OVERRIDE
                                                                            ///< modes.
    ZES_OC_MODE_FIXED = 3,                                                  ///< Overclocking fixed Mode - In this mode, hardware will disable most
                                                                            ///< frequency throttling and lock the frequency and voltage at the
                                                                            ///< specified overclock values. This mode disables OVERRIDE and
                                                                            ///< INTERPOLATIVE modes. This mode can damage the part, most of the
                                                                            ///< protections are disabled on this mode.
    ZES_OC_MODE_FORCE_UINT32 = 0x7fffffff

} zes_oc_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclocking properties
/// 
/// @details
///     - Provides all the overclocking capabilities and properties supported by
///       the device for the frequency domain.
///     - [DEPRECATED] No longer supported.
typedef struct _zes_oc_capabilities_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t isOcSupported;                                                ///< [out] Indicates if any overclocking features are supported on this
                                                                            ///< frequency domain.
    double maxFactoryDefaultFrequency;                                      ///< [out] Factory default non-overclock maximum frequency in Mhz.
    double maxFactoryDefaultVoltage;                                        ///< [out] Factory default voltage used for the non-overclock maximum
                                                                            ///< frequency in MHz.
    double maxOcFrequency;                                                  ///< [out] Maximum hardware overclocking frequency limit in Mhz.
    double minOcVoltageOffset;                                              ///< [out] The minimum voltage offset that can be applied to the
                                                                            ///< voltage/frequency curve. Note that this number can be negative.
    double maxOcVoltageOffset;                                              ///< [out] The maximum voltage offset that can be applied to the
                                                                            ///< voltage/frequency curve.
    double maxOcVoltage;                                                    ///< [out] The maximum overclock voltage that hardware supports.
    ze_bool_t isTjMaxSupported;                                             ///< [out] Indicates if the maximum temperature limit (TjMax) can be
                                                                            ///< changed for this frequency domain.
    ze_bool_t isIccMaxSupported;                                            ///< [out] Indicates if the maximum current (IccMax) can be changed for
                                                                            ///< this frequency domain.
    ze_bool_t isHighVoltModeCapable;                                        ///< [out] Indicates if this frequency domains supports a feature to set
                                                                            ///< very high voltages.
    ze_bool_t isHighVoltModeEnabled;                                        ///< [out] Indicates if very high voltages are permitted on this frequency
                                                                            ///< domain.
    ze_bool_t isExtendedModeSupported;                                      ///< [out] Indicates if the extended overclocking features are supported.
                                                                            ///< If this is supported, increments are on 1 Mhz basis.
    ze_bool_t isFixedModeSupported;                                         ///< [out] Indicates if the fixed mode is supported. In this mode, hardware
                                                                            ///< will disable most frequency throttling and lock the frequency and
                                                                            ///< voltage at the specified overclock values.

} zes_oc_capabilities_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of frequency domains
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFrequencyDomains(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_freq_handle_t* phFrequency                                          ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get frequency properties - available frequencies
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetProperties(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_freq_properties_t* pProperties                                      ///< [in,out] The frequency properties for the specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get available non-overclocked hardware clock frequencies for the
///        frequency domain
/// 
/// @details
///     - The list of available frequencies is returned in order of slowest to
///       fastest.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetAvailableClocks(
    zes_freq_handle_t hFrequency,                                           ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of frequencies.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of frequencies that are available.
                                                                            ///< if count is greater than the number of frequencies that are available,
                                                                            ///< then the driver shall update the value with the correct number of frequencies.
    double* phFrequency                                                     ///< [in,out][optional][range(0, *pCount)] array of frequencies in units of
                                                                            ///< MHz and sorted from slowest to fastest.
                                                                            ///< if count is less than the number of frequencies that are available,
                                                                            ///< then the driver shall only retrieve that number of frequencies.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current frequency limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLimits`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetRange(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_freq_range_t* pLimits                                               ///< [in,out] The range between which the hardware can operate for the
                                                                            ///< specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set frequency range between which the hardware can operate.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLimits`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencySetRange(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    const zes_freq_range_t* pLimits                                         ///< [in] The limits between which the hardware can operate for the
                                                                            ///< specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current frequency state - frequency request, actual frequency, TDP
///        limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetState(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_freq_state_t* pState                                                ///< [in,out] Frequency state for the specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get frequency throttle time
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pThrottleTime`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetThrottleTime(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_freq_throttle_time_t* pThrottleTime                                 ///< [in,out] Will contain a snapshot of the throttle time counters for the
                                                                            ///< specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the overclocking capabilities.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOcCapabilities`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetCapabilities(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_oc_capabilities_t* pOcCapabilities                                  ///< [in,out] Pointer to the capabilities structure.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current overclocking frequency target, if extended moded is
///        supported, will returned in 1 Mhz granularity.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCurrentOcFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see the `maxOcFrequency`, `maxOcVoltage`, `minOcVoltageOffset` and `maxOcVoltageOffset` members of ::zes_oc_capabilities_t).
///         + Requested voltage overclock is very high but the `isHighVoltModeEnabled` member of ::zes_oc_capabilities_t is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetFrequencyTarget(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double* pCurrentOcFrequency                                             ///< [out] Overclocking Frequency in MHz, if extended moded is supported,
                                                                            ///< will returned in 1 Mhz granularity, else, in multiples of 50 Mhz. This
                                                                            ///< cannot be greater than the `maxOcFrequency` member of
                                                                            ///< ::zes_oc_capabilities_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the current overclocking frequency target, if extended moded is
///        supported, can be set in 1 Mhz granularity.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see the `maxOcFrequency`, `maxOcVoltage`, `minOcVoltageOffset` and `maxOcVoltageOffset` members of ::zes_oc_capabilities_t).
///         + Requested voltage overclock is very high but the `isHighVoltModeEnabled` member of ::zes_oc_capabilities_t is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetFrequencyTarget(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double CurrentOcFrequency                                               ///< [in] Overclocking Frequency in MHz, if extended moded is supported, it
                                                                            ///< could be set in 1 Mhz granularity, else, in multiples of 50 Mhz. This
                                                                            ///< cannot be greater than the `maxOcFrequency` member of
                                                                            ///< ::zes_oc_capabilities_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current overclocking voltage settings.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCurrentVoltageTarget`
///         + `nullptr == pCurrentVoltageOffset`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see the `maxOcFrequency`, `maxOcVoltage`, `minOcVoltageOffset` and `maxOcVoltageOffset` members of ::zes_oc_capabilities_t).
///         + Requested voltage overclock is very high but the `isHighVoltModeEnabled` member of ::zes_oc_capabilities_t is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetVoltageTarget(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double* pCurrentVoltageTarget,                                          ///< [out] Overclock voltage in Volts. This cannot be greater than the
                                                                            ///< `maxOcVoltage` member of ::zes_oc_capabilities_t.
    double* pCurrentVoltageOffset                                           ///< [out] This voltage offset is applied to all points on the
                                                                            ///< voltage/frequency curve, including the new overclock voltageTarget.
                                                                            ///< Valid range is between the `minOcVoltageOffset` and
                                                                            ///< `maxOcVoltageOffset` members of ::zes_oc_capabilities_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the current overclocking voltage settings.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see the `maxOcFrequency`, `maxOcVoltage`, `minOcVoltageOffset` and `maxOcVoltageOffset` members of ::zes_oc_capabilities_t).
///         + Requested voltage overclock is very high but the `isHighVoltModeEnabled` member of ::zes_oc_capabilities_t is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetVoltageTarget(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double CurrentVoltageTarget,                                            ///< [in] Overclock voltage in Volts. This cannot be greater than the
                                                                            ///< `maxOcVoltage` member of ::zes_oc_capabilities_t.
    double CurrentVoltageOffset                                             ///< [in] This voltage offset is applied to all points on the
                                                                            ///< voltage/frequency curve, include the new overclock voltageTarget.
                                                                            ///< Valid range is between the `minOcVoltageOffset` and
                                                                            ///< `maxOcVoltageOffset` members of ::zes_oc_capabilities_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the current overclocking mode.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OC_MODE_FIXED < CurrentOcMode`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see the `maxOcFrequency`, `maxOcVoltage`, `minOcVoltageOffset` and `maxOcVoltageOffset` members of ::zes_oc_capabilities_t).
///         + Requested voltage overclock is very high but the `isHighVoltModeEnabled` member of ::zes_oc_capabilities_t is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetMode(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_oc_mode_t CurrentOcMode                                             ///< [in] Current Overclocking Mode ::zes_oc_mode_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current overclocking mode.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCurrentOcMode`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see the `maxOcFrequency`, `maxOcVoltage`, `minOcVoltageOffset` and `maxOcVoltageOffset` members of ::zes_oc_capabilities_t).
///         + Requested voltage overclock is very high but the `isHighVoltModeEnabled` member of ::zes_oc_capabilities_t is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetMode(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    zes_oc_mode_t* pCurrentOcMode                                           ///< [out] Current Overclocking Mode ::zes_oc_mode_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the maximum current limit setting.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOcIccMax`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + Capability the `isIccMaxSupported` member of ::zes_oc_capabilities_t is false for this frequency domain.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetIccMax(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double* pOcIccMax                                                       ///< [in,out] Will contain the maximum current limit in Amperes on
                                                                            ///< successful return.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change the maximum current limit setting.
/// 
/// @details
///     - Setting ocIccMax to 0.0 will return the value to the factory default.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The `isIccMaxSupported` member of ::zes_oc_capabilities_t is false for this frequency domain.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + The specified current limit is too low or too high.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetIccMax(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double ocIccMax                                                         ///< [in] The new maximum current limit in Amperes.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the maximum temperature limit setting.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOcTjMax`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetTjMax(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double* pOcTjMax                                                        ///< [in,out] Will contain the maximum temperature limit in degrees Celsius
                                                                            ///< on successful return.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change the maximum temperature limit setting.
/// 
/// @details
///     - Setting ocTjMax to 0.0 will return the value to the factory default.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (see the `isOcSupported` member of ::zes_oc_capabilities_t).
///         + The `isTjMaxSupported` member of ::zes_oc_capabilities_t is false for this frequency domain.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + The specified temperature limit is too high.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetTjMax(
    zes_freq_handle_t hFrequency,                                           ///< [in] Handle for the component.
    double ocTjMax                                                          ///< [in] The new maximum temperature limit in degrees Celsius.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region led
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief LED properties
typedef struct _zes_led_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                                                   ///< [out] Indicates if software can control the LED assuming the user has
                                                                            ///< permissions
    ze_bool_t haveRGB;                                                      ///< [out] Indicates if the LED is RGB capable

} zes_led_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief LED color
typedef struct _zes_led_color_t
{
    double red;                                                             ///< [in,out][range(0.0, 1.0)] The LED red value. On output, a value less
                                                                            ///< than 0.0 indicates that the color is not known.
    double green;                                                           ///< [in,out][range(0.0, 1.0)] The LED green value. On output, a value less
                                                                            ///< than 0.0 indicates that the color is not known.
    double blue;                                                            ///< [in,out][range(0.0, 1.0)] The LED blue value. On output, a value less
                                                                            ///< than 0.0 indicates that the color is not known.

} zes_led_color_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief LED state
typedef struct _zes_led_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t isOn;                                                         ///< [out] Indicates if the LED is on or off
    zes_led_color_t color;                                                  ///< [out] Color of the LED

} zes_led_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of LEDs
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumLeds(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_led_handle_t* phLed                                                 ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get LED properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedGetProperties(
    zes_led_handle_t hLed,                                                  ///< [in] Handle for the component.
    zes_led_properties_t* pProperties                                       ///< [in,out] Will contain the properties of the LED.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current state of a LED - on/off, color
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedGetState(
    zes_led_handle_t hLed,                                                  ///< [in] Handle for the component.
    zes_led_state_t* pState                                                 ///< [in,out] Will contain the current state of the LED.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Turn the LED on/off
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedSetState(
    zes_led_handle_t hLed,                                                  ///< [in] Handle for the component.
    ze_bool_t enable                                                        ///< [in] Set to TRUE to turn the LED on, FALSE to turn off.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the color of the LED
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pColor`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This LED doesn't not support color changes. See the `haveRGB` member of ::zes_led_properties_t.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedSetColor(
    zes_led_handle_t hLed,                                                  ///< [in] Handle for the component.
    const zes_led_color_t* pColor                                           ///< [in] New color of the LED.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Memory management
#if !defined(__GNUC__)
#pragma region memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Memory module types
typedef enum _zes_mem_type_t
{
    ZES_MEM_TYPE_HBM = 0,                                                   ///< HBM memory
    ZES_MEM_TYPE_DDR = 1,                                                   ///< DDR memory
    ZES_MEM_TYPE_DDR3 = 2,                                                  ///< DDR3 memory
    ZES_MEM_TYPE_DDR4 = 3,                                                  ///< DDR4 memory
    ZES_MEM_TYPE_DDR5 = 4,                                                  ///< DDR5 memory
    ZES_MEM_TYPE_LPDDR = 5,                                                 ///< LPDDR memory
    ZES_MEM_TYPE_LPDDR3 = 6,                                                ///< LPDDR3 memory
    ZES_MEM_TYPE_LPDDR4 = 7,                                                ///< LPDDR4 memory
    ZES_MEM_TYPE_LPDDR5 = 8,                                                ///< LPDDR5 memory
    ZES_MEM_TYPE_SRAM = 9,                                                  ///< SRAM memory
    ZES_MEM_TYPE_L1 = 10,                                                   ///< L1 cache
    ZES_MEM_TYPE_L3 = 11,                                                   ///< L3 cache
    ZES_MEM_TYPE_GRF = 12,                                                  ///< Execution unit register file
    ZES_MEM_TYPE_SLM = 13,                                                  ///< Execution unit shared local memory
    ZES_MEM_TYPE_GDDR4 = 14,                                                ///< GDDR4 memory
    ZES_MEM_TYPE_GDDR5 = 15,                                                ///< GDDR5 memory
    ZES_MEM_TYPE_GDDR5X = 16,                                               ///< GDDR5X memory
    ZES_MEM_TYPE_GDDR6 = 17,                                                ///< GDDR6 memory
    ZES_MEM_TYPE_GDDR6X = 18,                                               ///< GDDR6X memory
    ZES_MEM_TYPE_GDDR7 = 19,                                                ///< GDDR7 memory
    ZES_MEM_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory module location
typedef enum _zes_mem_loc_t
{
    ZES_MEM_LOC_SYSTEM = 0,                                                 ///< System memory
    ZES_MEM_LOC_DEVICE = 1,                                                 ///< On board local device memory
    ZES_MEM_LOC_FORCE_UINT32 = 0x7fffffff

} zes_mem_loc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory health
typedef enum _zes_mem_health_t
{
    ZES_MEM_HEALTH_UNKNOWN = 0,                                             ///< The memory health cannot be determined.
    ZES_MEM_HEALTH_OK = 1,                                                  ///< All memory channels are healthy.
    ZES_MEM_HEALTH_DEGRADED = 2,                                            ///< Excessive correctable errors have been detected on one or more
                                                                            ///< channels. Device should be reset.
    ZES_MEM_HEALTH_CRITICAL = 3,                                            ///< Operating with reduced memory to cover banks with too many
                                                                            ///< uncorrectable errors.
    ZES_MEM_HEALTH_REPLACE = 4,                                             ///< Device should be replaced due to excessive uncorrectable errors.
    ZES_MEM_HEALTH_FORCE_UINT32 = 0x7fffffff

} zes_mem_health_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory properties
typedef struct _zes_mem_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_mem_type_t type;                                                    ///< [out] The memory type
    ze_bool_t onSubdevice;                                                  ///< [out] True if this resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    zes_mem_loc_t location;                                                 ///< [out] Location of this memory (system, device)
    uint64_t physicalSize;                                                  ///< [out] Physical memory size in bytes. A value of 0 indicates that this
                                                                            ///< property is not known. However, a call to ::zesMemoryGetState() will
                                                                            ///< correctly return the total size of usable memory.
    int32_t busWidth;                                                       ///< [out] Width of the memory bus. A value of -1 means that this property
                                                                            ///< is unknown.
    int32_t numChannels;                                                    ///< [out] The number of memory channels. A value of -1 means that this
                                                                            ///< property is unknown.

} zes_mem_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory state - health, allocated
/// 
/// @details
///     - Percent allocation is given by 100 * (size - free / size.
///     - Percent free is given by 100 * free / size.
typedef struct _zes_mem_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_mem_health_t health;                                                ///< [out] Indicates the health of the memory
    uint64_t free;                                                          ///< [out] The free memory in bytes
    uint64_t size;                                                          ///< [out] The total allocatable memory in bytes (can be less than the
                                                                            ///< `physicalSize` member of ::zes_mem_properties_t)

} zes_mem_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory bandwidth
/// 
/// @details
///     - Percent bandwidth is calculated by taking two snapshots (s1, s2) and
///       using the equation: %bw = 10^6 * ((s2.readCounter - s1.readCounter) +
///       (s2.writeCounter - s1.writeCounter)) / (s2.maxBandwidth *
///       (s2.timestamp - s1.timestamp))
///     - Counter can roll over and rollover needs to be handled by comparing
///       the current read against the previous read
///     - Counter is a 32 byte transaction count, which means the calculated
///       delta (delta = current_value - previous_value or delta = 2^32 -
///       previous_value + current_value in case of rollover) needs to be
///       multiplied by 32 to get delta between samples in actual byte count
typedef struct _zes_mem_bandwidth_t
{
    uint64_t readCounter;                                                   ///< [out] Total bytes read from memory
    uint64_t writeCounter;                                                  ///< [out] Total bytes written to memory
    uint64_t maxBandwidth;                                                  ///< [out] Current maximum bandwidth in units of bytes/sec
    uint64_t timestamp;                                                     ///< [out] The timestamp in microseconds when these measurements were sampled.
                                                                            ///< This timestamp should only be used to calculate delta time between
                                                                            ///< snapshots of this structure.
                                                                            ///< Never take the delta of this timestamp with the timestamp from a
                                                                            ///< different structure since they are not guaranteed to have the same base.
                                                                            ///< The absolute value of the timestamp is only valid during within the
                                                                            ///< application and may be different on the next execution.

} zes_mem_bandwidth_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties for Memory bandwidth
/// 
/// @details
///     - Number of counter bits
///     - [DEPRECATED] No longer supported.
typedef struct _zes_mem_ext_bandwidth_t
{
    uint32_t memoryTimestampValidBits;                                      ///< [out] Returns the number of valid bits in the timestamp values

} zes_mem_ext_bandwidth_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of memory modules
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumMemoryModules(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_mem_handle_t* phMemory                                              ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetProperties(
    zes_mem_handle_t hMemory,                                               ///< [in] Handle for the component.
    zes_mem_properties_t* pProperties                                       ///< [in,out] Will contain memory properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory state - health, allocated
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetState(
    zes_mem_handle_t hMemory,                                               ///< [in] Handle for the component.
    zes_mem_state_t* pState                                                 ///< [in,out] Will contain the current health and allocated memory.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory bandwidth
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pBandwidth`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetBandwidth(
    zes_mem_handle_t hMemory,                                               ///< [in] Handle for the component.
    zes_mem_bandwidth_t* pBandwidth                                         ///< [in,out] Will contain the total number of bytes read from and written
                                                                            ///< to memory, as well as the current maximum bandwidth.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Performance factor
#if !defined(__GNUC__)
#pragma region performance
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Static information about a Performance Factor domain
typedef struct _zes_perf_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if this Performance Factor affects accelerators located on
                                                                            ///< a sub-device
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    zes_engine_type_flags_t engines;                                        ///< [out] Bitfield of accelerator engine types that are affected by this
                                                                            ///< Performance Factor.

} zes_perf_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handles to accelerator domains whose performance can be optimized
///        via a Performance Factor
/// 
/// @details
///     - A Performance Factor should be tuned for each workload.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumPerformanceFactorDomains(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_perf_handle_t* phPerf                                               ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties about a Performance Factor domain
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPerf`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorGetProperties(
    zes_perf_handle_t hPerf,                                                ///< [in] Handle for the Performance Factor domain.
    zes_perf_properties_t* pProperties                                      ///< [in,out] Will contain information about the specified Performance
                                                                            ///< Factor domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current Performance Factor for a given domain
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPerf`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pFactor`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorGetConfig(
    zes_perf_handle_t hPerf,                                                ///< [in] Handle for the Performance Factor domain.
    double* pFactor                                                         ///< [in,out] Will contain the actual Performance Factor being used by the
                                                                            ///< hardware (may not be the same as the requested Performance Factor).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change the performance factor for a domain
/// 
/// @details
///     - The Performance Factor is a number between 0 and 100.
///     - A Performance Factor is a hint to the hardware. Depending on the
///       hardware, the request may not be granted. Follow up this function with
///       a call to ::zesPerformanceFactorGetConfig() to determine the actual
///       factor being used by the hardware.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPerf`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorSetConfig(
    zes_perf_handle_t hPerf,                                                ///< [in] Handle for the Performance Factor domain.
    double factor                                                           ///< [in] The new Performance Factor.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Scheduler management
#if !defined(__GNUC__)
#pragma region power
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Power Domain
typedef enum _zes_power_domain_t
{
    ZES_POWER_DOMAIN_UNKNOWN = 0,                                           ///< The PUnit power domain level cannot be determined.
    ZES_POWER_DOMAIN_CARD = 1,                                              ///< The PUnit power domain is a card-level power domain.
    ZES_POWER_DOMAIN_PACKAGE = 2,                                           ///< The PUnit power domain is a package-level power domain.
    ZES_POWER_DOMAIN_STACK = 3,                                             ///< The PUnit power domain is a stack-level power domain.
    ZES_POWER_DOMAIN_MEMORY = 4,                                            ///< The PUnit power domain is a memory-level power domain.
    ZES_POWER_DOMAIN_GPU = 5,                                               ///< The PUnit power domain is a GPU-level power domain.
    ZES_POWER_DOMAIN_FORCE_UINT32 = 0x7fffffff

} zes_power_domain_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Power Level Type
typedef enum _zes_power_level_t
{
    ZES_POWER_LEVEL_UNKNOWN = 0,                                            ///< The PUnit power monitoring duration cannot be determined.
    ZES_POWER_LEVEL_SUSTAINED = 1,                                          ///< The PUnit determines effective power draw by computing a moving
                                                                            ///< average of the actual power draw over a time interval (longer than
                                                                            ///< BURST).
    ZES_POWER_LEVEL_BURST = 2,                                              ///< The PUnit determines effective power draw by computing a moving
                                                                            ///< average of the actual power draw over a time interval (longer than
                                                                            ///< PEAK).
    ZES_POWER_LEVEL_PEAK = 3,                                               ///< The PUnit determines effective power draw by computing a moving
                                                                            ///< average of the actual power draw over a very short time interval.
    ZES_POWER_LEVEL_INSTANTANEOUS = 4,                                      ///< The PUnit predicts effective power draw using the current device
                                                                            ///< configuration (frequency, voltage, etc...) & throttles proactively to
                                                                            ///< stay within the specified limit.
    ZES_POWER_LEVEL_FORCE_UINT32 = 0x7fffffff

} zes_power_level_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Power Source Type
typedef enum _zes_power_source_t
{
    ZES_POWER_SOURCE_ANY = 0,                                               ///< Limit active no matter whether the power source is mains powered or
                                                                            ///< battery powered.
    ZES_POWER_SOURCE_MAINS = 1,                                             ///< Limit active only when the device is mains powered.
    ZES_POWER_SOURCE_BATTERY = 2,                                           ///< Limit active only when the device is battery powered.
    ZES_POWER_SOURCE_FORCE_UINT32 = 0x7fffffff

} zes_power_source_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Limit Unit
typedef enum _zes_limit_unit_t
{
    ZES_LIMIT_UNIT_UNKNOWN = 0,                                             ///< The PUnit power monitoring unit cannot be determined.
    ZES_LIMIT_UNIT_CURRENT = 1,                                             ///< The limit is specified in milliamperes of current drawn.
    ZES_LIMIT_UNIT_POWER = 2,                                               ///< The limit is specified in milliwatts of power generated.
    ZES_LIMIT_UNIT_FORCE_UINT32 = 0x7fffffff

} zes_limit_unit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties related to device power settings
typedef struct _zes_power_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if this resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                                                   ///< [out] Software can change the power limits of this domain assuming the
                                                                            ///< user has permissions.
    ze_bool_t isEnergyThresholdSupported;                                   ///< [out] Indicates if this power domain supports the energy threshold
                                                                            ///< event (::ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED).
    int32_t defaultLimit;                                                   ///< [out] (Deprecated) The factory default TDP power limit of the part in
                                                                            ///< milliwatts. A value of -1 means that this is not known.
    int32_t minLimit;                                                       ///< [out] (Deprecated) The minimum power limit in milliwatts that can be
                                                                            ///< requested. A value of -1 means that this is not known.
    int32_t maxLimit;                                                       ///< [out] (Deprecated) The maximum power limit in milliwatts that can be
                                                                            ///< requested. A value of -1 means that this is not known.

} zes_power_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Energy counter snapshot
/// 
/// @details
///     - Average power is calculated by taking two snapshots (s1, s2) and using
///       the equation: PowerWatts = (s2.energy - s1.energy) / (s2.timestamp -
///       s1.timestamp)
typedef struct _zes_power_energy_counter_t
{
    uint64_t energy;                                                        ///< [out] The monotonic energy counter in microjoules.
    uint64_t timestamp;                                                     ///< [out] Microsecond timestamp when energy was captured.
                                                                            ///< This timestamp should only be used to calculate delta time between
                                                                            ///< snapshots of this structure.
                                                                            ///< Never take the delta of this timestamp with the timestamp from a
                                                                            ///< different structure since they are not guaranteed to have the same base.
                                                                            ///< The absolute value of the timestamp is only valid during within the
                                                                            ///< application and may be different on the next execution.

} zes_power_energy_counter_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sustained power limits
/// 
/// @details
///     - The power controller (Punit) will throttle the operating frequency if
///       the power averaged over a window (typically seconds) exceeds this
///       limit.
///     - [DEPRECATED] No longer supported.
typedef struct _zes_power_sustained_limit_t
{
    ze_bool_t enabled;                                                      ///< [in,out] indicates if the limit is enabled (true) or ignored (false)
    int32_t power;                                                          ///< [in,out] power limit in milliwatts
    int32_t interval;                                                       ///< [in,out] power averaging window (Tau) in milliseconds

} zes_power_sustained_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Burst power limit
/// 
/// @details
///     - The power controller (Punit) will throttle the operating frequency of
///       the device if the power averaged over a few milliseconds exceeds a
///       limit known as PL2. Typically PL2 > PL1 so that it permits the
///       frequency to burst higher for short periods than would be otherwise
///       permitted by PL1.
///     - [DEPRECATED] No longer supported.
typedef struct _zes_power_burst_limit_t
{
    ze_bool_t enabled;                                                      ///< [in,out] indicates if the limit is enabled (true) or ignored (false)
    int32_t power;                                                          ///< [in,out] power limit in milliwatts

} zes_power_burst_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Peak power limit
/// 
/// @details
///     - The power controller (Punit) will reactively/proactively throttle the
///       operating frequency of the device when the instantaneous/100usec power
///       exceeds this limit. The limit is known as PL4 or Psys. It expresses
///       the maximum power that can be drawn from the power supply.
///     - If this power limit is removed or set too high, the power supply will
///       generate an interrupt when it detects an overcurrent condition and the
///       power controller will throttle the device frequencies down to min. It
///       is thus better to tune the PL4 value in order to avoid such
///       excursions.
///     - [DEPRECATED] No longer supported.
typedef struct _zes_power_peak_limit_t
{
    int32_t powerAC;                                                        ///< [in,out] power limit in milliwatts for the AC power source.
    int32_t powerDC;                                                        ///< [in,out] power limit in milliwatts for the DC power source. On input,
                                                                            ///< this is ignored if the product does not have a battery. On output,
                                                                            ///< this will be -1 if the product does not have a battery.

} zes_power_peak_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Energy threshold
/// 
/// @details
///     - .
typedef struct _zes_energy_threshold_t
{
    ze_bool_t enable;                                                       ///< [in,out] Indicates if the energy threshold is enabled.
    double threshold;                                                       ///< [in,out] The energy threshold in Joules. Will be 0.0 if no threshold
                                                                            ///< has been set.
    uint32_t processId;                                                     ///< [in,out] The host process ID that set the energy threshold. Will be
                                                                            ///< 0xFFFFFFFF if no threshold has been set.

} zes_energy_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of power domains
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumPowerDomains(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_pwr_handle_t* phPower                                               ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of the PCIe card-level power
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phPower`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + The device does not provide access to card level power controls or telemetry. An invalid power domain handle will be returned in phPower.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetCardPowerDomain(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    zes_pwr_handle_t* phPower                                               ///< [in,out] power domain handle for the entire PCIe card.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties related to a power domain
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetProperties(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    zes_power_properties_t* pProperties                                     ///< [in,out] Structure that will contain property data.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get energy counter
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pEnergy`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetEnergyCounter(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    zes_power_energy_counter_t* pEnergy                                     ///< [in,out] Will contain the latest snapshot of the energy counter and
                                                                            ///< timestamp when the last counter value was measured.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get power limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] Use ::zesPowerGetLimitsExt.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetLimits(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    zes_power_sustained_limit_t* pSustained,                                ///< [in,out][optional] The sustained power limit. If this is null, the
                                                                            ///< current sustained power limits will not be returned.
    zes_power_burst_limit_t* pBurst,                                        ///< [in,out][optional] The burst power limit. If this is null, the current
                                                                            ///< peak power limits will not be returned.
    zes_power_peak_limit_t* pPeak                                           ///< [in,out][optional] The peak power limit. If this is null, the peak
                                                                            ///< power limits will not be returned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set power limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] Use ::zesPowerSetLimitsExt.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + The device is in use, meaning that the GPU is under Over clocking, applying power limits under overclocking is not supported.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetLimits(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    const zes_power_sustained_limit_t* pSustained,                          ///< [in][optional] The sustained power limit. If this is null, no changes
                                                                            ///< will be made to the sustained power limits.
    const zes_power_burst_limit_t* pBurst,                                  ///< [in][optional] The burst power limit. If this is null, no changes will
                                                                            ///< be made to the burst power limits.
    const zes_power_peak_limit_t* pPeak                                     ///< [in][optional] The peak power limit. If this is null, no changes will
                                                                            ///< be made to the peak power limits.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get energy threshold
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pThreshold`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Energy threshold not supported on this power domain (check the `isEnergyThresholdSupported` member of ::zes_power_properties_t).
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetEnergyThreshold(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    zes_energy_threshold_t* pThreshold                                      ///< [in,out] Returns information about the energy threshold setting -
                                                                            ///< enabled/energy threshold/process ID.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set energy threshold
/// 
/// @details
///     - An event ::ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED will be
///       generated when the delta energy consumed starting from this call
///       exceeds the specified threshold. Use the function
///       ::zesDeviceEventRegister() to start receiving the event.
///     - Only one running process can control the energy threshold at a given
///       time. If another process attempts to change the energy threshold, the
///       error ::ZE_RESULT_ERROR_NOT_AVAILABLE will be returned. The function
///       ::zesPowerGetEnergyThreshold() to determine the process ID currently
///       controlling this setting.
///     - Calling this function will remove any pending energy thresholds and
///       start counting from the time of this call.
///     - Once the energy threshold has been reached and the event generated,
///       the threshold is automatically removed. It is up to the application to
///       request a new threshold.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Energy threshold not supported on this power domain (check the `isEnergyThresholdSupported` member of ::zes_power_properties_t).
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Another running process has set the energy threshold.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetEnergyThreshold(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    double threshold                                                        ///< [in] The energy threshold to be set in joules.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region psu
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief PSU voltage status
typedef enum _zes_psu_voltage_status_t
{
    ZES_PSU_VOLTAGE_STATUS_UNKNOWN = 0,                                     ///< The status of the power supply voltage controllers cannot be
                                                                            ///< determined
    ZES_PSU_VOLTAGE_STATUS_NORMAL = 1,                                      ///< No unusual voltages have been detected
    ZES_PSU_VOLTAGE_STATUS_OVER = 2,                                        ///< Over-voltage has occurred
    ZES_PSU_VOLTAGE_STATUS_UNDER = 3,                                       ///< Under-voltage has occurred
    ZES_PSU_VOLTAGE_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_psu_voltage_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Static properties of the power supply
typedef struct _zes_psu_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t haveFan;                                                      ///< [out] True if the power supply has a fan
    int32_t ampLimit;                                                       ///< [out] The maximum electrical current in milliamperes that can be
                                                                            ///< drawn. A value of -1 indicates that this property cannot be
                                                                            ///< determined.

} zes_psu_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Dynamic state of the power supply
typedef struct _zes_psu_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_psu_voltage_status_t voltStatus;                                    ///< [out] The current PSU voltage status
    ze_bool_t fanFailed;                                                    ///< [out] Indicates if the fan has failed
    int32_t temperature;                                                    ///< [out] Read the current heatsink temperature in degrees Celsius. A
                                                                            ///< value of -1 indicates that this property cannot be determined.
    int32_t current;                                                        ///< [out] The amps being drawn in milliamperes. A value of -1 indicates
                                                                            ///< that this property cannot be determined.

} zes_psu_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of power supplies
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumPsus(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_psu_handle_t* phPsu                                                 ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get power supply properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPsu`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPsuGetProperties(
    zes_psu_handle_t hPsu,                                                  ///< [in] Handle for the component.
    zes_psu_properties_t* pProperties                                       ///< [in,out] Will contain the properties of the power supply.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current power supply state
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPsu`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPsuGetState(
    zes_psu_handle_t hPsu,                                                  ///< [in] Handle for the component.
    zes_psu_state_t* pState                                                 ///< [in,out] Will contain the current state of the power supply.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region ras
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error type
typedef enum _zes_ras_error_type_t
{
    ZES_RAS_ERROR_TYPE_CORRECTABLE = 0,                                     ///< Errors were corrected by hardware
    ZES_RAS_ERROR_TYPE_UNCORRECTABLE = 1,                                   ///< Error were not corrected
    ZES_RAS_ERROR_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error categories
typedef enum _zes_ras_error_cat_t
{
    ZES_RAS_ERROR_CAT_RESET = 0,                                            ///< The number of accelerator engine resets attempted by the driver
    ZES_RAS_ERROR_CAT_PROGRAMMING_ERRORS = 1,                               ///< The number of hardware exceptions generated by the way workloads have
                                                                            ///< programmed the hardware
    ZES_RAS_ERROR_CAT_DRIVER_ERRORS = 2,                                    ///< The number of low level driver communication errors have occurred
    ZES_RAS_ERROR_CAT_COMPUTE_ERRORS = 3,                                   ///< The number of errors that have occurred in the compute accelerator
                                                                            ///< hardware
    ZES_RAS_ERROR_CAT_NON_COMPUTE_ERRORS = 4,                               ///< The number of errors that have occurred in the fixed-function
                                                                            ///< accelerator hardware
    ZES_RAS_ERROR_CAT_CACHE_ERRORS = 5,                                     ///< The number of errors that have occurred in caches (L1/L3/register
                                                                            ///< file/shared local memory/sampler)
    ZES_RAS_ERROR_CAT_DISPLAY_ERRORS = 6,                                   ///< The number of errors that have occurred in the display
    ZES_RAS_ERROR_CAT_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_cat_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_RAS_ERROR_CATEGORY_COUNT
/// @brief The maximum number of categories
#define ZES_MAX_RAS_ERROR_CATEGORY_COUNT  7
#endif // ZES_MAX_RAS_ERROR_CATEGORY_COUNT

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS properties
typedef struct _zes_ras_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_ras_error_type_t type;                                              ///< [out] The type of RAS error
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device

} zes_ras_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error details
typedef struct _zes_ras_state_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t category[ZES_MAX_RAS_ERROR_CATEGORY_COUNT];                    ///< [in][out] Breakdown of error by category

} zes_ras_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error configuration - thresholds used for triggering RAS events
///        (::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS,
///        ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS)
/// 
/// @details
///     - The driver maintains a total counter which is updated every time a
///       hardware block covered by the corresponding RAS error set notifies
///       that an error has occurred. When this total count goes above the
///       totalThreshold specified below, a RAS event is triggered.
///     - The driver also maintains a counter for each category of RAS error
///       (see ::zes_ras_state_t for a breakdown). Each time a hardware block of
///       that category notifies that an error has occurred, that corresponding
///       category counter is updated. When it goes above the threshold
///       specified in detailedThresholds, a RAS event is triggered.
typedef struct _zes_ras_config_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t totalThreshold;                                                ///< [in,out] If the total RAS errors exceeds this threshold, the event
                                                                            ///< will be triggered. A value of 0ULL disables triggering the event based
                                                                            ///< on the total counter.
    zes_ras_state_t detailedThresholds;                                     ///< [in,out] If the RAS errors for each category exceed the threshold for
                                                                            ///< that category, the event will be triggered. A value of 0ULL will
                                                                            ///< disable an event being triggered for that category.

} zes_ras_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of all RAS error sets on a device
/// 
/// @details
///     - A RAS error set is a collection of RAS error counters of a given type
///       (correctable/uncorrectable) from hardware blocks contained within a
///       sub-device or within the device.
///     - A device without sub-devices will typically return two handles, one
///       for correctable errors sets and one for uncorrectable error sets.
///     - A device with sub-devices will return RAS error sets for each
///       sub-device and possibly RAS error sets for hardware blocks outside the
///       sub-devices.
///     - If the function completes successfully but pCount is set to 0, RAS
///       features are not available/enabled on this device.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumRasErrorSets(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_ras_handle_t* phRas                                                 ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get RAS properties of a given RAS error set - this enables discovery
///        of the type of RAS error set (correctable/uncorrectable) and if
///        located on a sub-device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetProperties(
    zes_ras_handle_t hRas,                                                  ///< [in] Handle for the component.
    zes_ras_properties_t* pProperties                                       ///< [in,out] Structure describing RAS properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get RAS error thresholds that control when RAS events are generated
/// 
/// @details
///     - The driver maintains counters for all RAS error sets and error
///       categories. Events are generated when errors occur. The configuration
///       enables setting thresholds to limit when events are sent.
///     - When a particular RAS correctable error counter exceeds the configured
///       threshold, the event ::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS will
///       be triggered.
///     - When a particular RAS uncorrectable error counter exceeds the
///       configured threshold, the event
///       ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS will be triggered.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetConfig(
    zes_ras_handle_t hRas,                                                  ///< [in] Handle for the component.
    zes_ras_config_t* pConfig                                               ///< [in,out] Will be populed with the current RAS configuration -
                                                                            ///< thresholds used to trigger events
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set RAS error thresholds that control when RAS events are generated
/// 
/// @details
///     - The driver maintains counters for all RAS error sets and error
///       categories. Events are generated when errors occur. The configuration
///       enables setting thresholds to limit when events are sent.
///     - When a particular RAS correctable error counter exceeds the specified
///       threshold, the event ::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS will
///       be generated.
///     - When a particular RAS uncorrectable error counter exceeds the
///       specified threshold, the event
///       ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS will be generated.
///     - Call ::zesRasGetState() and set the clear flag to true to restart
///       event generation once counters have exceeded thresholds.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Another running process is controlling these settings.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + Don't have permissions to set thresholds.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasSetConfig(
    zes_ras_handle_t hRas,                                                  ///< [in] Handle for the component.
    const zes_ras_config_t* pConfig                                         ///< [in] Change the RAS configuration - thresholds used to trigger events
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current value of RAS error counters for a particular error set
/// 
/// @details
///     - Clearing errors will affect other threads/applications - the counter
///       values will start from zero.
///     - Clearing errors requires write permissions.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + Don't have permissions to clear error counters.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetState(
    zes_ras_handle_t hRas,                                                  ///< [in] Handle for the component.
    ze_bool_t clear,                                                        ///< [in] Set to 1 to clear the counters of this type
    zes_ras_state_t* pState                                                 ///< [in,out] Breakdown of where errors have occurred
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Scheduler management
#if !defined(__GNUC__)
#pragma region scheduler
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Scheduler mode
typedef enum _zes_sched_mode_t
{
    ZES_SCHED_MODE_TIMEOUT = 0,                                             ///< Multiple applications or contexts are submitting work to the hardware.
                                                                            ///< When higher priority work arrives, the scheduler attempts to pause the
                                                                            ///< current executing work within some timeout interval, then submits the
                                                                            ///< other work.
    ZES_SCHED_MODE_TIMESLICE = 1,                                           ///< The scheduler attempts to fairly timeslice hardware execution time
                                                                            ///< between multiple contexts submitting work to the hardware
                                                                            ///< concurrently.
    ZES_SCHED_MODE_EXCLUSIVE = 2,                                           ///< Any application or context can run indefinitely on the hardware
                                                                            ///< without being preempted or terminated. All pending work for other
                                                                            ///< contexts must wait until the running context completes with no further
                                                                            ///< submitted work.
    ZES_SCHED_MODE_COMPUTE_UNIT_DEBUG = 3,                                  ///< [DEPRECATED] No longer supported.
    ZES_SCHED_MODE_FORCE_UINT32 = 0x7fffffff

} zes_sched_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties related to scheduler component
typedef struct _zes_sched_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t onSubdevice;                                                  ///< [out] True if this resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                                                   ///< [out] Software can change the scheduler component configuration
                                                                            ///< assuming the user has permissions.
    zes_engine_type_flags_t engines;                                        ///< [out] Bitfield of accelerator engine types that are managed by this
                                                                            ///< scheduler component. Note that there can be more than one scheduler
                                                                            ///< component for the same type of accelerator engine.
    uint32_t supportedModes;                                                ///< [out] Bitfield of scheduler modes that can be configured for this
                                                                            ///< scheduler component (bitfield of 1<<::zes_sched_mode_t).

} zes_sched_properties_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_SCHED_WATCHDOG_DISABLE
/// @brief Disable forward progress guard timeout.
#define ZES_SCHED_WATCHDOG_DISABLE  (~(0ULL))
#endif // ZES_SCHED_WATCHDOG_DISABLE

///////////////////////////////////////////////////////////////////////////////
/// @brief Configuration for timeout scheduler mode (::ZES_SCHED_MODE_TIMEOUT)
typedef struct _zes_sched_timeout_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t watchdogTimeout;                                               ///< [in,out] The maximum time in microseconds that the scheduler will wait
                                                                            ///< for a batch of work submitted to a hardware engine to complete or to
                                                                            ///< be preempted so as to run another context.
                                                                            ///< If this time is exceeded, the hardware engine is reset and the context terminated.
                                                                            ///< If set to ::ZES_SCHED_WATCHDOG_DISABLE, a running workload can run as
                                                                            ///< long as it wants without being terminated, but preemption attempts to
                                                                            ///< run other contexts are permitted but not enforced.

} zes_sched_timeout_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Configuration for timeslice scheduler mode
///        (::ZES_SCHED_MODE_TIMESLICE)
typedef struct _zes_sched_timeslice_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t interval;                                                      ///< [in,out] The average interval in microseconds that a submission for a
                                                                            ///< context will run on a hardware engine before being preempted out to
                                                                            ///< run a pending submission for another context.
    uint64_t yieldTimeout;                                                  ///< [in,out] The maximum time in microseconds that the scheduler will wait
                                                                            ///< to preempt a workload running on an engine before deciding to reset
                                                                            ///< the hardware engine and terminating the associated context.

} zes_sched_timeslice_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns handles to scheduler components.
/// 
/// @details
///     - Each scheduler component manages the distribution of work across one
///       or more accelerator engines.
///     - If an application wishes to change the scheduler behavior for all
///       accelerator engines of a specific type (e.g. compute), it should
///       select all the handles where the `engines` member
///       ::zes_sched_properties_t contains that type.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumSchedulers(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_sched_handle_t* phScheduler                                         ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties related to a scheduler component
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetProperties(
    zes_sched_handle_t hScheduler,                                          ///< [in] Handle for the component.
    zes_sched_properties_t* pProperties                                     ///< [in,out] Structure that will contain property data.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current scheduling mode in effect on a scheduler component.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMode`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetCurrentMode(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    zes_sched_mode_t* pMode                                                 ///< [in,out] Will contain the current scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get scheduler config for mode ::ZES_SCHED_MODE_TIMEOUT
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetTimeoutModeProperties(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    ze_bool_t getDefaults,                                                  ///< [in] If TRUE, the driver will return the system default properties for
                                                                            ///< this mode, otherwise it will return the current properties.
    zes_sched_timeout_properties_t* pConfig                                 ///< [in,out] Will contain the current parameters for this mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get scheduler config for mode ::ZES_SCHED_MODE_TIMESLICE
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetTimesliceModeProperties(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    ze_bool_t getDefaults,                                                  ///< [in] If TRUE, the driver will return the system default properties for
                                                                            ///< this mode, otherwise it will return the current properties.
    zes_sched_timeslice_properties_t* pConfig                               ///< [in,out] Will contain the current parameters for this mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_TIMEOUT or update scheduler
///        mode parameters if already running in this mode.
/// 
/// @details
///     - This mode is optimized for multiple applications or contexts
///       submitting work to the hardware. When higher priority work arrives,
///       the scheduler attempts to pause the current executing work within some
///       timeout interval, then submits the other work.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetTimeoutMode(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    zes_sched_timeout_properties_t* pProperties,                            ///< [in] The properties to use when configurating this mode.
    ze_bool_t* pNeedReload                                                  ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                                            ///< apply the new scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_TIMESLICE or update
///        scheduler mode parameters if already running in this mode.
/// 
/// @details
///     - This mode is optimized to provide fair sharing of hardware execution
///       time between multiple contexts submitting work to the hardware
///       concurrently.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetTimesliceMode(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    zes_sched_timeslice_properties_t* pProperties,                          ///< [in] The properties to use when configurating this mode.
    ze_bool_t* pNeedReload                                                  ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                                            ///< apply the new scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_EXCLUSIVE
/// 
/// @details
///     - This mode is optimized for single application/context use-cases. It
///       permits a context to run indefinitely on the hardware without being
///       preempted or terminated. All pending work for other contexts must wait
///       until the running context completes with no further submitted work.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetExclusiveMode(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    ze_bool_t* pNeedReload                                                  ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                                            ///< apply the new scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_COMPUTE_UNIT_DEBUG
/// 
/// @details
///     - This is a special mode that must ben enabled when debugging an
///       application that uses this device e.g. using the Level0 Debug API.
///     - It ensures that only one command queue can execute work on the
///       hardware at a given time. Work is permitted to run as long as needed
///       without enforcing any scheduler fairness policies.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - [DEPRECATED] No longer supported.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetComputeUnitDebugMode(
    zes_sched_handle_t hScheduler,                                          ///< [in] Sysman handle for the component.
    ze_bool_t* pNeedReload                                                  ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                                            ///< apply the new scheduler mode.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Standby domains
#if !defined(__GNUC__)
#pragma region standby
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Standby hardware components
typedef enum _zes_standby_type_t
{
    ZES_STANDBY_TYPE_GLOBAL = 0,                                            ///< Control the overall standby policy of the device/sub-device
    ZES_STANDBY_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_standby_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Standby hardware component properties
typedef struct _zes_standby_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_standby_type_t type;                                                ///< [out] Which standby hardware component this controls
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device

} zes_standby_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Standby promotion modes
typedef enum _zes_standby_promo_mode_t
{
    ZES_STANDBY_PROMO_MODE_DEFAULT = 0,                                     ///< Best compromise between performance and energy savings.
    ZES_STANDBY_PROMO_MODE_NEVER = 1,                                       ///< The device/component will never shutdown. This can improve performance
                                                                            ///< but uses more energy.
    ZES_STANDBY_PROMO_MODE_FORCE_UINT32 = 0x7fffffff

} zes_standby_promo_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of standby controls
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumStandbyDomains(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_standby_handle_t* phStandby                                         ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get standby hardware component properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hStandby`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbyGetProperties(
    zes_standby_handle_t hStandby,                                          ///< [in] Handle for the component.
    zes_standby_properties_t* pProperties                                   ///< [in,out] Will contain the standby hardware properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current standby promotion mode
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hStandby`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMode`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbyGetMode(
    zes_standby_handle_t hStandby,                                          ///< [in] Handle for the component.
    zes_standby_promo_mode_t* pMode                                         ///< [in,out] Will contain the current standby mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set standby promotion mode
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hStandby`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_STANDBY_PROMO_MODE_NEVER < mode`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbySetMode(
    zes_standby_handle_t hStandby,                                          ///< [in] Handle for the component.
    zes_standby_promo_mode_t mode                                           ///< [in] New standby mode.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region temperature
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature sensors
typedef enum _zes_temp_sensors_t
{
    ZES_TEMP_SENSORS_GLOBAL = 0,                                            ///< The maximum temperature across all device sensors
    ZES_TEMP_SENSORS_GPU = 1,                                               ///< The maximum temperature across all sensors in the GPU
    ZES_TEMP_SENSORS_MEMORY = 2,                                            ///< The maximum temperature across all sensors in the local memory
    ZES_TEMP_SENSORS_GLOBAL_MIN = 3,                                        ///< The minimum temperature across all device sensors
    ZES_TEMP_SENSORS_GPU_MIN = 4,                                           ///< The minimum temperature across all sensors in the GPU
    ZES_TEMP_SENSORS_MEMORY_MIN = 5,                                        ///< The minimum temperature across all sensors in the local device memory
    ZES_TEMP_SENSORS_GPU_BOARD = 6,                                         ///< The maximum temperature across all sensors in the GPU Board
    ZES_TEMP_SENSORS_GPU_BOARD_MIN = 7,                                     ///< The minimum temperature across all sensors in the GPU Board
    ZES_TEMP_SENSORS_FORCE_UINT32 = 0x7fffffff

} zes_temp_sensors_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature sensor properties
typedef struct _zes_temp_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_temp_sensors_t type;                                                ///< [out] Which part of the device the temperature sensor measures
    ze_bool_t onSubdevice;                                                  ///< [out] True if the resource is located on a sub-device; false means
                                                                            ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    double maxTemperature;                                                  ///< [out] Will contain the maximum temperature for the specific device in
                                                                            ///< degrees Celsius.
    ze_bool_t isCriticalTempSupported;                                      ///< [out] Indicates if the critical temperature event
                                                                            ///< ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL is supported
    ze_bool_t isThreshold1Supported;                                        ///< [out] Indicates if the temperature threshold 1 event
                                                                            ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 is supported
    ze_bool_t isThreshold2Supported;                                        ///< [out] Indicates if the temperature threshold 2 event
                                                                            ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 is supported

} zes_temp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature sensor threshold
typedef struct _zes_temp_threshold_t
{
    ze_bool_t enableLowToHigh;                                              ///< [in,out] Trigger an event when the temperature crosses from below the
                                                                            ///< threshold to above.
    ze_bool_t enableHighToLow;                                              ///< [in,out] Trigger an event when the temperature crosses from above the
                                                                            ///< threshold to below.
    double threshold;                                                       ///< [in,out] The threshold in degrees Celsius.

} zes_temp_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature configuration - which events should be triggered and the
///        trigger conditions.
typedef struct _zes_temp_config_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t enableCritical;                                               ///< [in,out] Indicates if event ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL should
                                                                            ///< be triggered by the driver.
    zes_temp_threshold_t threshold1;                                        ///< [in,out] Configuration controlling if and when event
                                                                            ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 should be triggered by the
                                                                            ///< driver.
    zes_temp_threshold_t threshold2;                                        ///< [in,out] Configuration controlling if and when event
                                                                            ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 should be triggered by the
                                                                            ///< driver.

} zes_temp_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of temperature sensors
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumTemperatureSensors(
    zes_device_handle_t hDevice,                                            ///< [in] Sysman handle of the device.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of components of this type.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of components of this type that are available.
                                                                            ///< if count is greater than the number of components of this type that
                                                                            ///< are available, then the driver shall update the value with the correct
                                                                            ///< number of components.
    zes_temp_handle_t* phTemperature                                        ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                                            ///< this type.
                                                                            ///< if count is less than the number of components of this type that are
                                                                            ///< available, then the driver shall only retrieve that number of
                                                                            ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get temperature sensor properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetProperties(
    zes_temp_handle_t hTemperature,                                         ///< [in] Handle for the component.
    zes_temp_properties_t* pProperties                                      ///< [in,out] Will contain the temperature sensor properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get temperature configuration for this sensor - which events are
///        triggered and the trigger conditions
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Temperature thresholds are not supported on this temperature sensor. Generally this is only supported for temperature sensor ::ZES_TEMP_SENSORS_GLOBAL.
///         + One or both of the thresholds is not supported. Check the `isThreshold1Supported` and `isThreshold2Supported` members of ::zes_temp_properties_t.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetConfig(
    zes_temp_handle_t hTemperature,                                         ///< [in] Handle for the component.
    zes_temp_config_t* pConfig                                              ///< [in,out] Returns current configuration.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set temperature configuration for this sensor - indicates which events
///        are triggered and the trigger conditions
/// 
/// @details
///     - Events ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL will be triggered when
///       temperature reaches the critical range. Use the function
///       ::zesDeviceEventRegister() to start receiving this event.
///     - Events ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 and
///       ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 will be generated when
///       temperature cross the thresholds set using this function. Use the
///       function ::zesDeviceEventRegister() to start receiving these events.
///     - Only one running process can set the temperature configuration at a
///       time. If another process attempts to change the configuration, the
///       error ::ZE_RESULT_ERROR_NOT_AVAILABLE will be returned. The function
///       ::zesTemperatureGetConfig() will return the process ID currently
///       controlling these settings.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Temperature thresholds are not supported on this temperature sensor. Generally they are only supported for temperature sensor ::ZES_TEMP_SENSORS_GLOBAL.
///         + Enabling the critical temperature event is not supported. Check the `isCriticalTempSupported` member of ::zes_temp_properties_t.
///         + One or both of the thresholds is not supported. Check the `isThreshold1Supported` and `isThreshold2Supported` members of ::zes_temp_properties_t.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Another running process is controlling these settings.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + One or both the thresholds is above TjMax (see ::zesFrequencyOcGetTjMax()). Temperature thresholds must be below this value.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureSetConfig(
    zes_temp_handle_t hTemperature,                                         ///< [in] Handle for the component.
    const zes_temp_config_t* pConfig                                        ///< [in] New configuration.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the temperature from a specified sensor
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pTemperature`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetState(
    zes_temp_handle_t hTemperature,                                         ///< [in] Handle for the component.
    double* pTemperature                                                    ///< [in,out] Will contain the temperature read from the specified sensor
                                                                            ///< in degrees Celsius.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Sysman Extension APIs for Power Limits
#if !defined(__GNUC__)
#pragma region powerLimits
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_POWER_LIMITS_EXT_NAME
/// @brief Power Limits Extension Name
#define ZES_POWER_LIMITS_EXT_NAME  "ZES_extension_power_limits"
#endif // ZES_POWER_LIMITS_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Power Limits Extension Version(s)
typedef enum _zes_power_limits_ext_version_t
{
    ZES_POWER_LIMITS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),             ///< version 1.0
    ZES_POWER_LIMITS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),         ///< latest known version
    ZES_POWER_LIMITS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} zes_power_limits_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device power/current limit descriptor.
typedef struct _zes_power_limit_ext_desc_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_power_level_t level;                                                ///< [in,out] duration type over which the power draw is measured, i.e.
                                                                            ///< sustained, burst, peak, or critical.
    zes_power_source_t source;                                              ///< [out] source of power used by the system, i.e. AC or DC.
    zes_limit_unit_t limitUnit;                                             ///< [out] unit used for specifying limit, i.e. current units (milliamps)
                                                                            ///< or power units (milliwatts).
    ze_bool_t enabledStateLocked;                                           ///< [out] indicates if the power limit state (enabled/ignored) can be set
                                                                            ///< (false) or is locked (true).
    ze_bool_t enabled;                                                      ///< [in,out] indicates if the limit is enabled (true) or ignored (false).
                                                                            ///< If enabledStateIsLocked is True, this value is ignored.
    ze_bool_t intervalValueLocked;                                          ///< [out] indicates if the interval can be modified (false) or is fixed
                                                                            ///< (true).
    int32_t interval;                                                       ///< [in,out] power averaging window in milliseconds. If
                                                                            ///< intervalValueLocked is true, this value is ignored.
    ze_bool_t limitValueLocked;                                             ///< [out] indicates if the limit can be set (false) or if the limit is
                                                                            ///< fixed (true).
    int32_t limit;                                                          ///< [in,out] limit value. If limitValueLocked is true, this value is
                                                                            ///< ignored. The value should be provided in the unit specified by
                                                                            ///< limitUnit.

} zes_power_limit_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties related to device power settings
/// 
/// @details
///     - This structure may be returned from ::zesPowerGetProperties via the
///       `pNext` member of ::zes_power_properties_t.
///     - This structure may also be returned from ::zesPowerGetProperties via
///       the `pNext` member of ::zes_power_ext_properties_t
///     - Used for determining the power domain level, i.e. card-level v/s
///       package-level v/s stack-level & the factory default power limits.
typedef struct _zes_power_ext_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_power_domain_t domain;                                              ///< [out] domain that the power limit belongs to.
    zes_power_limit_ext_desc_t* defaultLimit;                               ///< [out] the factory default limit of the part.

} zes_power_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get power limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - This function returns all the power limits associated with the
///       supplied power domain.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetLimitsExt(
    zes_pwr_handle_t hPower,                                                ///< [in] Power domain handle instance.
    uint32_t* pCount,                                                       ///< [in,out] Pointer to the number of power limit descriptors. If count is
                                                                            ///< zero, then the driver shall update the value with the total number of
                                                                            ///< components of this type that are available. If count is greater than
                                                                            ///< the number of components of this type that are available, then the
                                                                            ///< driver shall update the value with the correct number of components.
    zes_power_limit_ext_desc_t* pSustained                                  ///< [in,out][optional][range(0, *pCount)] Array of query results for power
                                                                            ///< limit descriptors. If count is less than the number of components of
                                                                            ///< this type that are available, then the driver shall only retrieve that
                                                                            ///< number of components.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set power limits
/// 
/// @details
///     - The application can only modify unlocked members of the limit
///       descriptors returned by ::zesPowerGetLimitsExt.
///     - Not all the limits returned by ::zesPowerGetLimitsExt need to be
///       supplied to this function.
///     - Limits do not have to be supplied in the same order as returned by
///       ::zesPowerGetLimitsExt.
///     - The same limit can be supplied multiple times. Limits are applied in
///       the order in which they are supplied.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + The device is in use, meaning that the GPU is under Over clocking, applying power limits under overclocking is not supported.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetLimitsExt(
    zes_pwr_handle_t hPower,                                                ///< [in] Handle for the component.
    uint32_t* pCount,                                                       ///< [in] Pointer to the number of power limit descriptors.
    zes_power_limit_ext_desc_t* pSustained                                  ///< [in][optional][range(0, *pCount)] Array of power limit descriptors.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Sysman Extension APIs for Engine Activity
#if !defined(__GNUC__)
#pragma region engineActivity
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_ENGINE_ACTIVITY_EXT_NAME
/// @brief Engine Activity Extension Name
#define ZES_ENGINE_ACTIVITY_EXT_NAME  "ZES_extension_engine_activity"
#endif // ZES_ENGINE_ACTIVITY_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Engine Activity Extension Version(s)
typedef enum _zes_engine_activity_ext_version_t
{
    ZES_ENGINE_ACTIVITY_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),          ///< version 1.0
    ZES_ENGINE_ACTIVITY_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),      ///< latest known version
    ZES_ENGINE_ACTIVITY_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} zes_engine_activity_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties related to Engine Groups
/// 
/// @details
///     - This structure may be passed to ::zesEngineGetProperties by having the
///       pNext member of ::zes_engine_properties_t point at this struct.
///     - Used for SRIOV per Virtual Function device utilization by
///       ::zes_engine_group_t
typedef struct _zes_engine_ext_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t countOfVirtualFunctionInstance;                                ///< [out] Number of Virtual Function(VF) instances associated with engine
                                                                            ///< to monitor the utilization of hardware across all Virtual Function
                                                                            ///< from a Physical Function (PF) instance.
                                                                            ///< These VF-by-VF views should provide engine group and individual engine
                                                                            ///< level granularity.
                                                                            ///< This count represents the number of VF instances that are actively
                                                                            ///< using the resource represented by the engine handle.

} zes_engine_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get activity stats for Physical Function (PF) and each Virtual
///        Function (VF) associated with engine group.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEngine`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE - "Engine activity extension is not supported in the environment."
ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetActivityExt(
    zes_engine_handle_t hEngine,                                            ///< [in] Handle for the component.
    uint32_t* pCount,                                                       ///< [in,out] Pointer to the number of VF engine stats descriptors.
                                                                            ///<  - if count is zero, the driver shall update the value with the total
                                                                            ///< number of engine stats available.
                                                                            ///<  - if count is greater than the total number of engine stats
                                                                            ///< available, the driver shall update the value with the correct number
                                                                            ///< of engine stats available.
                                                                            ///<  - The count returned is the sum of number of VF instances currently
                                                                            ///< available and the PF instance.
    zes_engine_stats_t* pStats                                              ///< [in,out][optional][range(0, *pCount)] array of engine group activity counters.
                                                                            ///<  - if count is less than the total number of engine stats available,
                                                                            ///< then driver shall only retrieve that number of stats.
                                                                            ///<  - the implementation shall populate the vector with engine stat for
                                                                            ///< PF at index 0 of the vector followed by user provided pCount-1 number
                                                                            ///< of VF engine stats.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Sysman Extension APIs for RAS Get State and Clear State
#if !defined(__GNUC__)
#pragma region rasState
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_RAS_GET_STATE_EXP_NAME
/// @brief RAS Get State Extension Name
#define ZES_RAS_GET_STATE_EXP_NAME  "ZES_extension_ras_state"
#endif // ZES_RAS_GET_STATE_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS Get State Extension Version(s)
typedef enum _zes_ras_state_exp_version_t
{
    ZES_RAS_STATE_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                ///< version 1.0
    ZES_RAS_STATE_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),            ///< latest known version
    ZES_RAS_STATE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zes_ras_state_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error categories
typedef enum _zes_ras_error_category_exp_t
{
    ZES_RAS_ERROR_CATEGORY_EXP_RESET = 0,                                   ///< The number of accelerator engine resets attempted by the driver
    ZES_RAS_ERROR_CATEGORY_EXP_PROGRAMMING_ERRORS = 1,                      ///< The number of hardware exceptions generated by the way workloads have
                                                                            ///< programmed the hardware
    ZES_RAS_ERROR_CATEGORY_EXP_DRIVER_ERRORS = 2,                           ///< The number of low level driver communication errors have occurred
    ZES_RAS_ERROR_CATEGORY_EXP_COMPUTE_ERRORS = 3,                          ///< The number of errors that have occurred in the compute accelerator
                                                                            ///< hardware
    ZES_RAS_ERROR_CATEGORY_EXP_NON_COMPUTE_ERRORS = 4,                      ///< The number of errors that have occurred in the fixed-function
                                                                            ///< accelerator hardware
    ZES_RAS_ERROR_CATEGORY_EXP_CACHE_ERRORS = 5,                            ///< The number of errors that have occurred in caches (L1/L3/register
                                                                            ///< file/shared local memory/sampler)
    ZES_RAS_ERROR_CATEGORY_EXP_DISPLAY_ERRORS = 6,                          ///< The number of errors that have occurred in the display
    ZES_RAS_ERROR_CATEGORY_EXP_MEMORY_ERRORS = 7,                           ///< The number of errors that have occurred in Memory
    ZES_RAS_ERROR_CATEGORY_EXP_SCALE_ERRORS = 8,                            ///< The number of errors that have occurred in Scale Fabric
    ZES_RAS_ERROR_CATEGORY_EXP_L3FABRIC_ERRORS = 9,                         ///< The number of errors that have occurred in L3 Fabric
    ZES_RAS_ERROR_CATEGORY_EXP_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_category_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension structure for providing RAS error counters for different
///        error sets
typedef struct _zes_ras_state_exp_t
{
    zes_ras_error_category_exp_t category;                                  ///< [out] category for which error counter is provided.
    uint64_t errorCounter;                                                  ///< [out] Current value of RAS counter for specific error category.

} zes_ras_state_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ras Get State
/// 
/// @details
///     - This function retrieves error counters for different RAS error
///       categories.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetStateExp(
    zes_ras_handle_t hRas,                                                  ///< [in] Handle for the component.
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of RAS state structures that can be retrieved.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of error categories for which state can be retrieved.
                                                                            ///< if count is greater than the number of RAS states available, then the
                                                                            ///< driver shall update the value with the correct number of RAS states available.
    zes_ras_state_exp_t* pState                                             ///< [in,out][optional][range(0, *pCount)] array of query results for RAS
                                                                            ///< error states for different categories.
                                                                            ///< if count is less than the number of RAS states available, then driver
                                                                            ///< shall only retrieve that number of RAS states.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Ras Clear State
/// 
/// @details
///     - This function clears error counters for a RAS error category.
///     - Clearing errors will affect other threads/applications - the counter
///       values will start from zero.
///     - Clearing errors requires write permissions.
///     - The application should not call this function from simultaneous
///       threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_RAS_ERROR_CATEGORY_EXP_L3FABRIC_ERRORS < category`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + Don't have permissions to clear error counters.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasClearStateExp(
    zes_ras_handle_t hRas,                                                  ///< [in] Handle for the component.
    zes_ras_error_category_exp_t category                                   ///< [in] category for which error counter is to be cleared.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Sysman Extension APIs for Memory State
#if !defined(__GNUC__)
#pragma region memPageOfflineState
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MEM_PAGE_OFFLINE_STATE_EXP_NAME
/// @brief Memory State Extension Name
#define ZES_MEM_PAGE_OFFLINE_STATE_EXP_NAME  "ZES_extension_mem_state"
#endif // ZES_MEM_PAGE_OFFLINE_STATE_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory State Extension Version(s)
typedef enum _zes_mem_page_offline_state_exp_version_t
{
    ZES_MEM_PAGE_OFFLINE_STATE_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),   ///< version 1.0
    ZES_MEM_PAGE_OFFLINE_STATE_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),   ///< latest known version
    ZES_MEM_PAGE_OFFLINE_STATE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zes_mem_page_offline_state_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties for Memory State
/// 
/// @details
///     - This structure may be returned from ::zesMemoryGetState via the
///       `pNext` member of ::zes_mem_state_t
///     - These additional parameters get Memory Page Offline Metrics
typedef struct _zes_mem_page_offline_state_exp_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t memoryPageOffline;                                             ///< [out] Returns the number of Memory Pages Offline
    uint32_t maxMemoryPageOffline;                                          ///< [out] Returns the Allowed Memory Pages Offline

} zes_mem_page_offline_state_exp_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Sysman Extension APIs for Memory Timestamp Valid Bits
#if !defined(__GNUC__)
#pragma region memoryTimestampValidBits
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MEMORY_TIMESTAMP_VALID_BITS_EXP_NAME
/// @brief Memory Timestamp Valid Bits Extension Name
#define ZES_MEMORY_TIMESTAMP_VALID_BITS_EXP_NAME  "ZES_extension_mem_timestamp_valid_bits"
#endif // ZES_MEMORY_TIMESTAMP_VALID_BITS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory Timestamp Valid Bits Extension Version(s)
typedef enum _zes_mem_timestamp_bits_exp_version_t
{
    ZES_MEM_TIMESTAMP_BITS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),       ///< version 1.0
    ZES_MEM_TIMESTAMP_BITS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),   ///< latest known version
    ZES_MEM_TIMESTAMP_BITS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zes_mem_timestamp_bits_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties for reporting valid bit count for memory
///        timestamp value
/// 
/// @details
///     - This structure may be returned from ::zesMemoryGetProperties via the
///       `pNext` member of ::zes_mem_properties_t.
///     - Used for denoting number of valid bits in the timestamp value returned
///       in ::zes_mem_bandwidth_t.
typedef struct _zes_mem_timestamp_bits_exp_t
{
    uint32_t memoryTimestampValidBits;                                      ///< [out] Returns the number of valid bits in the timestamp values

} zes_mem_timestamp_bits_exp_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Sysman Extension APIs for Power Domain Properties
#if !defined(__GNUC__)
#pragma region powerDomainProperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_POWER_DOMAIN_PROPERTIES_EXP_NAME
/// @brief Power Domain Properties Name
#define ZES_POWER_DOMAIN_PROPERTIES_EXP_NAME  "ZES_extension_power_domain_properties"
#endif // ZES_POWER_DOMAIN_PROPERTIES_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Power Domain Properties Extension Version(s)
typedef enum _zes_power_domain_properties_exp_version_t
{
    ZES_POWER_DOMAIN_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZES_POWER_DOMAIN_PROPERTIES_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZES_POWER_DOMAIN_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zes_power_domain_properties_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension structure for providing power domain information associated
///        with a power handle
/// 
/// @details
///     - This structure may be returned from ::zesPowerGetProperties via the
///       `pNext` member of ::zes_power_properties_t.
///     - Used for associating a power handle with a power domain.
typedef struct _zes_power_domain_exp_properties_t
{
    zes_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zes_power_domain_t powerDomain;                                         ///< [out] Power domain associated with the power handle.

} zes_power_domain_exp_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZES_API_H