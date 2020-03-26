#ifndef _MYD_VSC_H_
#define _MYD_VSC_H_

#include <linux/ioctl.h>
#define IOC_MYD_TYPE	        't'
#define IOC_MYD_MIN_NR	        30
#define IOC_MYD_WRITE	        _IOWR(IOC_MYD_TYPE, 30, struct __myd_ion_data)
#define IOC_MYD_READ	        _IOWR(IOC_MYD_TYPE, 31, struct __myd_ion_data)
#define RAW_MYD_READ            _IOWR(IOC_MYD_TYPE, 32, struct __myd_r_data)
#define IOC_MYD_MAX_NR	        32

typedef int ion_user_handle_t;

typedef struct __myd_ion_data {
  ion_user_handle_t handle;
  size_t len;
  unsigned long phys_addr;
  unsigned long cpu_addr;
  unsigned int timeout_msec;
}myd_write_data_t, myd_read_data_t;

typedef struct __myd_r_data {
    void *data;
    int ss;
    unsigned int timeout_msec;
}read_data_t;

#endif

