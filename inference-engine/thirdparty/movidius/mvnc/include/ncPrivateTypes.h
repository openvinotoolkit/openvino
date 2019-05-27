/*
* Copyright 2018-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

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

typedef enum {
    NC_OPTION_CLASS0 = 0,
    NC_OPTION_CLASS1 = 1,
    NC_OPTION_CLASS2 = 2,
    NC_OPTION_CLASS3 = 3,
} ncOptionClass_t;

typedef enum {
    NC_FIFO_HWC = 0, // row major - channel minor, for RGB image: RGB00, RGB01, RGB02,...
                     // all RGB pixels by row
    NC_FIFO_CHW = 1, // channel major - column minor (planar), for RGB image:
                     // R01R02R03...G01G02G03...B01B02B03...
                     // all Red rows..all Green rows..all Blue rows
    NC_FIFO_HCW = 2, // row major - column minor (interleaved), for RGB image:
                     // R00R01..R0k.., G00G01..G0k.., B00B01..B0k.., R10R11..R1k..
                     // 1st Red row, 1st Green row, 1st Blue Rrw, 2nd Red row..
    NC_FIFO_CWH = 3, // channel major - row minor, for RGB image:
                     // R00R10R20... G00G10G20...B00B10B20...
                     // all Red columns, all Green columns, all blue columns
    NC_FIFO_WCH = 4, // column major - row minor; for RGB image:
                     // R00R10..Rk0.., G00G10..Gk0.., B00B10..Bk0.., R01R11..Rk1..
                     // 1st Red col, 1st Green col, 1st blue col, 2nd Red col...
    NC_FIFO_WHC = 5, // column major - channle minor, for RGB image: RGB00, RGB10, RGB20...
                     // all RGB pixels by col...
} ncFifoLayout_t;

struct _devicePrivate_t {
    int throttle_happened;
    float *thermal_stats;
    char *dev_addr;     // Device USB address as returned by usb_
    char *dev_addr_booted;
    char *dev_file;     // Device filename in /dev directory
    char *optimisation_list;
    XLinkHandler_t *xlink;
    struct _devicePrivate_t *next;  // Next device in chain
    struct _graphPrivate_t *graphs; // List of associated graphs
    struct _fifoPrivate_t *fifos;   // List of associated fifos
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
    uint32_t deviceFreq;
    uint8_t* profilingBuffer;
    size_t   receivedData;
    wd_context watchdog_ctx;
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

struct _fifoPrivate_t {
    ncFifoType_t type;
    ncFifoLayout_t graphLayout;
    int consumer_cnt;
    uint32_t id;
    streamId_t streamId;
    struct ncTensorDescriptor_t graph_tensor_desc;
    struct ncTensorDescriptor_t host_tensor_desc;
    struct _devicePrivate_t *dev;
    struct _fifoPrivate_t *next;
    char name[NC_MAX_NAME_SIZE];
    struct _userParamPrivate_t *user_param_in;  //used for write fifo
    struct _userParamPrivate_t *user_param_out; //used for read fifo
    int host_tensor_desc_set;
    int write_count;
    int consumed_by_graph;
    int num_elements;
    int api_read_element;
    int consumers_remaining;
    int datasize;
    int timeout_msec;
    pthread_mutex_t fifo_mutex;
    ncFifoState_t state;
    void* output_data;
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
