// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Includes
// ----------------------------------------------------------------------------

#ifndef _NC_PRIVATE_TYPES_H_
#define _NC_PRIVATE_TYPES_H_

#if (defined(_WIN32) || defined(_WIN64))
#include "win_pthread.h"
#else
#include <pthread.h>
#endif

#include <mvnc.h>
#include "ncCommPrivate.h"
#include "XLinkPublicDefines.h"
#include "watchdog.h"

#define GRAPH_OPTION_BASE   1000
#define DEVICE_OPTION_BASE  2000
#define OPTION_CLASS_SIZE   100

typedef enum {
    NC_OP_ACCESS_READ_ONLY  = 0,
    NC_OP_ACCESS_READ_WRITE = 1,
    NC_OP_ACCESS_LAST       = 2,   // Last configuration option available for graph
} ncOptionAccess_t;

struct _devicePrivate_t {
    int throttle_happened;
    float *thermal_stats;
    XLinkProtocol_t protocol;
    char *dev_addr;     // Device USB address as returned by usb_
    XLinkProtocol_t protocol_booted;
    char *dev_addr_booted;
    char *dev_file;     // Device filename in /dev directory
    char *optimisation_list;
    XLinkHandler_t *xlink;
    struct _devicePrivate_t *next;  // Next device in chain
    struct _graphPrivate_t *graphs; // List of associated graphs
    streamId_t device_mon_stream_id;
    streamId_t graph_monitor_stream_id;
    streamId_t printf_over_xlink_stream_id;
    int        printf_over_xlink_conn_fd;
    pthread_t  printf_over_xlink_thr;
    int        printf_over_xlink_thr_valid;
    pthread_mutex_t dev_data_m;
    pthread_mutex_t dev_stream_m;
    pthread_mutex_t graph_stream_m;
    deviceCapabilities_t dev_attr;
    ncDeviceState_t state;
    uint32_t device_id;
    WdDeviceHndl_t* watchdog_device;
    int wd_interval;
};

extern devicePrivate_t *devices;

struct _userParamPrivate_t {
    void *data;
    struct _userParamPrivate_t *next;
};
struct _graphPrivate_t {
    uint32_t id;
    uint32_t blob_version[2];
    int started;
    int batch_size;
    int executors_number;
    int input_count;
    int output_count;
    struct ncTensorDescriptor_t input_tensor_desc;
    struct ncTensorDescriptor_t output_tensor_desc;
    unsigned nstages;
    int timingsCount;
    struct _devicePrivate_t *dev;
    struct _graphPrivate_t *next;
    size_t aux_buffer_size;
    char *aux_buffer;
    char *debug_buffer;
    char name[NC_MAX_NAME_SIZE];
    float *time_taken;
    streamId_t graph_stream_id;
    ncGraphState_t state;
};

#if (!defined(_WIN32) && !defined(_WIN64))
#define PACKED(name) struct __attribute__((packed)) name
#else
#define PACKED( __Declaration__ ) __pragma( pack(push, 1) ) struct __Declaration__ __pragma( pack(pop) )
#endif


#define EI_NIDENT 16

PACKED(ElfN_Ehdr
{
    uint8_t  e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint32_t e_entry;
    uint32_t e_phoff;
    uint32_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};)

PACKED(blob_header_v2
{
    uint32_t magic_number;              // =???, not used
    uint32_t file_size;                 // size of blob? not used
    uint32_t blob_ver_major;            // =???, not used
    uint32_t blob_ver_minor;            // =???, not used
    uint32_t bss_mem_size;
    uint32_t mode;
    uint32_t stage_section_offset;
    uint32_t buffer_section_offset;     // must be aligned by 16 bytes
    uint32_t relocation_section_offset;
};)

PACKED(stage_section_header_v2
{
    uint32_t stage_count;
    uint32_t stage_section_size;    // not used
    uint32_t input_size;
    uint32_t output_size;
    uint32_t batch_size;    
};)

#endif
