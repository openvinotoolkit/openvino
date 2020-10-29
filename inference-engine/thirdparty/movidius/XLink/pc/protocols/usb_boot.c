// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <sys/stat.h>
#if (defined(_WIN32) || defined(_WIN64) )
#include "win_usb.h"
#include "win_time.h"
#include "win_pthread.h"
#else
#include <unistd.h>
#include <getopt.h>
#include <libusb.h>
#include <pthread.h>
#endif
#include "usb_boot.h"

#include "XLinkStringUtils.h"
#include "XLinkPublicDefines.h"


#define DEFAULT_VID                 0x03E7

#define DEFAULT_WRITE_TIMEOUT       2000
#define DEFAULT_CONNECT_TIMEOUT     20000
#define DEFAULT_SEND_FILE_TIMEOUT   10000
#define USB1_CHUNKSZ                64

/*
 * ADDRESS_BUFF_SIZE 35 = 4*7+7.
 * '255-' x 7 (also gives us nul-terminator for last entry)
 * 7 => to add "-maXXXX"
 */
#define ADDRESS_BUFF_SIZE           35

#define OPEN_DEV_ERROR_MESSAGE_LENGTH 128

static unsigned int bulk_chunklen = DEFAULT_CHUNKSZ;
static int write_timeout = DEFAULT_WRITE_TIMEOUT;
static int connect_timeout = DEFAULT_CONNECT_TIMEOUT;
static int initialized;

typedef struct {
    int pid;
    char name[10];
} deviceBootInfo_t;

static deviceBootInfo_t supportedDevices[] = {
    {
        .pid = 0x2150,
        .name = "ma2450"
    },
    {
        .pid = 0x2485,
        .name = "ma2480"
    },
    {
        //To support the case where the port name change, or it's already booted
        .pid = DEFAULT_OPENPID,
        .name = ""
    }
};
// for now we'll only use the loglevel for usb boot. can bring it into
// the rest of usblink later
// use same levels as mvnc_loglevel for now
int usb_loglevel = 0;
#if (defined(_WIN32) || defined(_WIN64) )
void initialize_usb_boot()
{
    if (initialized == 0)
    {
        usb_init();
    }
    // We sanitize the situation by trying to reset the devices that have been left open
    initialized = 1;
}
#else
void __attribute__((constructor)) usb_library_load()
{
    initialized = !libusb_init(NULL);
}

void __attribute__((destructor)) usb_library_unload()
{
    if(initialized)
        libusb_exit(NULL);
}
#endif

typedef struct timespec highres_time_t;

static inline void highres_gettime(highres_time_t *ptr) {
    clock_gettime(CLOCK_REALTIME, ptr);
}

static inline double highres_elapsed_ms(highres_time_t *start, highres_time_t *end) {
    struct timespec temp;
    if((end->tv_nsec - start->tv_nsec) < 0) {
        temp.tv_sec = end->tv_sec - start->tv_sec - 1;
        temp.tv_nsec = 1000000000 + end->tv_nsec-start->tv_nsec;
    } else {
        temp.tv_sec = end->tv_sec - start->tv_sec;
        temp.tv_nsec = end->tv_nsec - start->tv_nsec;
    }
    return (double)(temp.tv_sec * 1000) + (((double)temp.tv_nsec) * 0.000001);
}

static const char *get_pid_name(int pid)
{
    int n = sizeof(supportedDevices)/sizeof(supportedDevices[0]);
    int i;

    for (i = 0; i < n; i++)
    {
        if (supportedDevices[i].pid == pid)
            return supportedDevices[i].name;
    }

    if(usb_loglevel)
        fprintf(stderr, "%s(): Error pid:=%i not supported\n", __func__, pid);

    return NULL;
}

const char * usb_get_pid_name(int pid)
{
    return get_pid_name(pid);
}

int get_pid_by_name(const char* name)
{
    char* p = strchr(name, '-');
    if (p == NULL) {
        if (usb_loglevel) {
            fprintf(stderr, "%s(): Error name (%s) not supported\n", __func__, name);
        }

        return -1;
    }
    p++; //advance to point to the name
    int i;
    int n = sizeof(supportedDevices)/sizeof(supportedDevices[0]);

    for (i = 0; i < n; i++)
    {
        if (strcmp(supportedDevices[i].name, p) == 0)
            return supportedDevices[i].pid;
    }
    return -1;
}

static int is_pid_supported(int pid)
{
    int n = sizeof(supportedDevices)/sizeof(supportedDevices[0]);
    int i;
    for (i = 0; i < n; i++) {
        if (supportedDevices[i].pid == pid)
            return 1;
    }
    return 0;
}

static int isMyriadDevice(const int idVendor, const int idProduct) {
    // Device is Myriad and pid supported
    if (idVendor == DEFAULT_VID && is_pid_supported(idProduct) == 1)
        return 1;
    // Device is Myriad and device booted
    if (idVendor == DEFAULT_OPENVID && idProduct == DEFAULT_OPENPID)
        return 1;
    return 0;
}

static int isBootedMyriadDevice(const int idVendor, const int idProduct) {
    // Device is Myriad, booted device pid
    if (idVendor == DEFAULT_VID && idProduct == DEFAULT_OPENPID) {
        return 1;
    }
    return 0;
}
static int isNotBootedMyriadDevice(const int idVendor, const int idProduct) {
    // Device is Myriad, pid supported and it's is not booted device
    if (idVendor == DEFAULT_VID && is_pid_supported(idProduct) == 1
        && idProduct != DEFAULT_OPENPID) {
        return 1;
    }
    return 0;
}

#if (!defined(_WIN32) && !defined(_WIN64) )
static const char *gen_addr(libusb_device *dev, int pid)
{
    static char buff[ADDRESS_BUFF_SIZE];
    uint8_t pnums[7];
    int pnum_cnt, i;
    char *p;

    pnum_cnt = libusb_get_port_numbers(dev, pnums, 7);
    if (pnum_cnt == LIBUSB_ERROR_OVERFLOW) {
        // shouldn't happen!
        mv_strcpy(buff, ADDRESS_BUFF_SIZE, "<error>");
        return buff;
    }
    p = buff;

#ifdef XLINK_USE_BUS
    uint8_t bus = libusb_get_bus_number(dev);
    p += snprintf(p, sizeof(buff), "%u.", bus);
#endif

    for (i = 0; i < pnum_cnt - 1; i++)
        p += snprintf(p, sizeof(buff),"%u.", pnums[i]);

    p += snprintf(p, sizeof(buff),"%u", pnums[i]);
    const char* dev_name = get_pid_name(pid);

    if (dev_name != NULL) {
        snprintf(p, sizeof(buff),"-%s", dev_name);
    } else {
        mv_strcpy(buff, ADDRESS_BUFF_SIZE,"<error>");
        return buff;
    }

    return buff;
}

static pthread_mutex_t globalMutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Find usb device address
 * @param input_addr  Device name (address) which would be returned. If not empty, we will try to
 *                  find device with this name
 *
 * @details
 * Find any device (device = 0):
 * <br> 1) Any myriad device:                    vid = AUTO_VID & pid = AUTO_PID
 * <br> 2) Any not booted myriad device:         vid = AUTO_VID & pid = AUTO_UNBOOTED_PID
 * <br> 3) Any booted myriad device:             vid = AUTO_VID & pid = DEFAULT_OPENPID
 * <br> 4) Specific Myriad 2 or Myriad X device: vid = AUTO_VID & pid = DEFAULT_UNBOOTPID_2485 or DEFAULT_UNBOOTPID_2150
 * <br><br> Find specific device (device != 0):
 * <br> device arg should be not null, search by addr (name) and return device struct
 *
 * @note
 * Index can be used to iterate through all connected myriad devices and save their names.
 * It will loop only over suitable devices specified by vid and pid
 */
usbBootError_t usb_find_device_with_bcd(unsigned idx, char *input_addr,
                                        unsigned addrsize, void **device, int vid, int pid, uint16_t* bcdusb) {
    if (pthread_mutex_lock(&globalMutex)) {
        fprintf(stderr, "Mutex lock failed\n");
        return USB_BOOT_ERROR;
    }
    int searchByName = 0;
    static libusb_device **devs = NULL;
    libusb_device *dev = NULL;
    struct libusb_device_descriptor desc;
    int count = 0;
    size_t i;
    int res;

    if (!initialized) {
        if (usb_loglevel)
            fprintf(stderr, "Library has not been initialized when loaded\n");
        if (pthread_mutex_unlock(&globalMutex)) {
            fprintf(stderr, "Mutex unlock failed\n");
        }
        return USB_BOOT_ERROR;
    }

    if (strlen(input_addr) > 1) {
        searchByName = 1;
    }

    // Update device list if empty or if indx 0
    if (!devs || idx == 0) {
        if (devs) {
            libusb_free_device_list(devs, 1);
            devs = 0;
        }
        if ((res = libusb_get_device_list(NULL, &devs)) < 0) {
            if (usb_loglevel)
                fprintf(stderr, "Unable to get USB device list: %s\n", libusb_strerror(res));
            if (pthread_mutex_unlock(&globalMutex)) {
                fprintf(stderr, "Mutex unlock failed\n");
            }
            return USB_BOOT_ERROR;
        }
    }

    // Loop over all usb devices, increase count only if myriad device
    i = 0;
    while ((dev = devs[i++]) != NULL) {
        if ((res = libusb_get_device_descriptor(dev, &desc)) < 0) {
            if (usb_loglevel)
                fprintf(stderr, "Unable to get USB device descriptor: %s\n", libusb_strerror(res));
            continue;
        }

        // If found device have the same id and vid as input
        if ( (desc.idVendor == vid && desc.idProduct == pid)
             // Any myriad device
             || (vid == AUTO_VID && pid == AUTO_PID
                 && isMyriadDevice(desc.idVendor, desc.idProduct))
             // Any not booted myriad device
             || (vid == AUTO_VID && (pid == AUTO_UNBOOTED_PID)
                 && isNotBootedMyriadDevice(desc.idVendor, desc.idProduct))
             // Any not booted with specific pid
             || (vid == AUTO_VID && pid == desc.idProduct
                 && isNotBootedMyriadDevice(desc.idVendor, desc.idProduct))
             // Any booted device
             || (vid == AUTO_VID && pid == DEFAULT_OPENPID
                 && isBootedMyriadDevice(desc.idVendor, desc.idProduct)) )
        {
            if (device) {
                const char *dev_addr = gen_addr(dev, get_pid_by_name(input_addr));
                if (!strcmp(dev_addr, input_addr)) {
                    if (usb_loglevel > 1) {
                        fprintf(stderr, "Found Address: %s - VID/PID %04x:%04x\n",
                                input_addr, desc.idVendor, desc.idProduct);
                    }

                    libusb_ref_device(dev);
                    libusb_free_device_list(devs, 1);
                    if (bcdusb)
                        *bcdusb = desc.bcdUSB;
                    *device = dev;
                    devs = 0;

                    if (pthread_mutex_unlock(&globalMutex)) {
                        fprintf(stderr, "Mutex unlock failed\n");
                    }
                    return USB_BOOT_SUCCESS;
                }
            } else if (searchByName) {
                const char *dev_addr = gen_addr(dev, desc.idProduct);
                // If the same add as input
                if (!strcmp(dev_addr, input_addr)) {
                    if (usb_loglevel > 1) {
                        fprintf(stderr, "Found Address: %s - VID/PID %04x:%04x\n",
                                input_addr, desc.idVendor, desc.idProduct);
                    }

                    if (pthread_mutex_unlock(&globalMutex)) {
                        fprintf(stderr, "Mutex unlock failed\n");
                    }
                    return USB_BOOT_SUCCESS;
                }
            } else if (idx == count) {
                const char *caddr = gen_addr(dev, desc.idProduct);
                if (usb_loglevel > 1)
                    fprintf(stderr, "Device %d Address: %s - VID/PID %04x:%04x\n",
                            idx, caddr, desc.idVendor, desc.idProduct);
                mv_strncpy(input_addr, addrsize, caddr, addrsize - 1);
                if (pthread_mutex_unlock(&globalMutex)) {
                    fprintf(stderr, "Mutex unlock failed\n");
                }
                return USB_BOOT_SUCCESS;
            }
            count++;
        }
    }
    libusb_free_device_list(devs, 1);
    devs = 0;
    if (pthread_mutex_unlock(&globalMutex)) {
        fprintf(stderr, "Mutex unlock failed\n");
    }
    return USB_BOOT_DEVICE_NOT_FOUND;
}
#endif

#if (defined(_WIN32) || defined(_WIN64) )
usbBootError_t usb_find_device(unsigned idx, char *addr, unsigned addrsize, void **device, int vid, int pid)
{
    if (!addr)
        return USB_BOOT_ERROR;
    int specificDevice = 0;
    if (strlen(addr) > 1)
        specificDevice = 1;

    // TODO There is no global mutex as in linux version
    int res;
    // 2 => vid
    // 2 => pid
    // '255-' x 7 (also gives us nul-terminator for last entry)
    // 7 => to add "-maXXXX"
    static uint8_t devs[128][2 + 2 + 4 * 7 + 7] = { 0 };//to store ven_id,dev_id;
    static int devs_cnt = 0;
    int count = 0;
    size_t i;

    if(!initialized)
    {
        if(usb_loglevel)
            fprintf(stderr, "Library has not been initialized when loaded\n");
        return USB_BOOT_ERROR;
    }
    if (devs_cnt == 0 || idx == 0) {
        devs_cnt = 0;
        if (((res = usb_list_devices(vid, pid, devs)) < 0)) {
            if (usb_loglevel)
                fprintf(stderr, "Unable to get USB device list: %s\n", libusb_strerror(res));
            return USB_BOOT_ERROR;
        }
        devs_cnt = res;
    } else {
        res = devs_cnt;
    }
    i = 0;

    while (res-- > 0) {
        int idVendor = (int)(devs[res][0] << 8 | devs[res][1]);
        int idProduct = (devs[res][2] << 8 | devs[res][3]);

        // If found device have the same id and vid as input
        if ( (idVendor == vid && idProduct == pid)
                // Any myriad device
                || (vid == AUTO_VID && pid == AUTO_PID
                        && isMyriadDevice(idVendor, idProduct))
                // Any unbooted myriad device
                || (vid == AUTO_VID && (pid == AUTO_UNBOOTED_PID)
                        && isNotBootedMyriadDevice(idVendor, idProduct))
                // Any unbooted with same pid
                || (vid == AUTO_VID && pid == idProduct
                        && isNotBootedMyriadDevice(idVendor, idProduct))
                // Any booted device
                || (vid == AUTO_VID && pid == DEFAULT_OPENPID
                        && isBootedMyriadDevice(idVendor, idProduct)) )
        {
            if (device) {
                const char *caddr = &devs[res][4];
                if (strncmp(addr, caddr, XLINK_MAX_NAME_SIZE) == 0)
                {
                    if (usb_loglevel > 1)
                        fprintf(stderr, "Found Address: %s - VID/PID %04x:%04x\n", caddr, (int)(devs[res][0] << 8 | devs[res][1]), (int)(devs[res][2] << 8 | devs[res][3]));
                    *device = enumerate_usb_device(vid, pid, caddr, 0);
                    devs_cnt = 0;
                    return USB_BOOT_SUCCESS;
                }
            }
            else if (specificDevice) {
                const char *caddr = &devs[res][4];
                if (strncmp(addr, caddr, XLINK_MAX_NAME_SIZE) == 0)
                {
                    if (usb_loglevel > 1)
                        fprintf(stderr, "Found Address: %s - VID/PID %04x:%04x\n", caddr, (int)(devs[res][0] << 8 | devs[res][1]), (int)(devs[res][2] << 8 | devs[res][3]));
                    return USB_BOOT_SUCCESS;
                }
            }
            else if (idx == count)
            {
                const char *caddr = &devs[res][4];
                if (usb_loglevel > 1)
                    fprintf(stderr, "Device %d Address: %s - VID/PID %04x:%04x\n", idx, caddr, (int)(devs[res][0] << 8 | devs[res][1]), (int)(devs[res][2] << 8 | devs[res][3]));
                mv_strncpy(addr, addrsize, caddr, addrsize - 1);
                return USB_BOOT_SUCCESS;
            }
            count++;
        }
    }
    devs_cnt = 0;
    return USB_BOOT_DEVICE_NOT_FOUND;
}
#endif


#if (!defined(_WIN32) && !defined(_WIN64) )
static libusb_device_handle *usb_open_device(libusb_device *dev, uint8_t *endpoint, char *err_string_buff, int err_max_len)
{
    struct libusb_config_descriptor *cdesc;
    const struct libusb_interface_descriptor *ifdesc;
    libusb_device_handle *h = NULL;
    int res, i;

    if((res = libusb_open(dev, &h)) < 0)
    {
        snprintf(err_string_buff, err_max_len, "cannot open device: %s\n", libusb_strerror(res));
        return 0;
    }
    if((res = libusb_set_configuration(h, 1)) < 0)
    {
        snprintf(err_string_buff, err_max_len, "setting config 1 failed: %s\n", libusb_strerror(res));
        libusb_close(h);
        return 0;
    }
    if((res = libusb_claim_interface(h, 0)) < 0)
    {
        snprintf(err_string_buff, err_max_len, "claiming interface 0 failed: %s\n", libusb_strerror(res));
        libusb_close(h);
        return 0;
    }
    if((res = libusb_get_config_descriptor(dev, 0, &cdesc)) < 0)
    {
        snprintf(err_string_buff, err_max_len, "Unable to get USB config descriptor: %s\n", libusb_strerror(res));
        libusb_close(h);
        return 0;
    }
    ifdesc = cdesc->interface->altsetting;
    for(i=0; i<ifdesc->bNumEndpoints; i++)
    {
        if(usb_loglevel > 1)
            fprintf(stderr, "Found EP 0x%02x : max packet size is %u bytes\n",
                    ifdesc->endpoint[i].bEndpointAddress, ifdesc->endpoint[i].wMaxPacketSize);
        if((ifdesc->endpoint[i].bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) != LIBUSB_TRANSFER_TYPE_BULK)
            continue;
        if( !(ifdesc->endpoint[i].bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK) )
        {
            *endpoint = ifdesc->endpoint[i].bEndpointAddress;
            bulk_chunklen = ifdesc->endpoint[i].wMaxPacketSize;
            libusb_free_config_descriptor(cdesc);
            return h;
        }
    }
    libusb_free_config_descriptor(cdesc);
    mv_strcpy(err_string_buff, OPEN_DEV_ERROR_MESSAGE_LENGTH,
              "Unable to find BULK OUT endpoint\n");
    libusb_close(h);
    return 0;
}
#endif
// timeout: -1 = no (infinite) timeout, 0 = must happen immediately


#if (!defined(_WIN32) && !defined(_WIN64) )
static int wait_findopen(const char *device_address, int timeout, libusb_device **dev, libusb_device_handle **devh, uint8_t *endpoint,uint16_t* bcdusb)
#else
static int wait_findopen(const char *device_address, int timeout, libusb_device **dev, libusb_device_handle **devh, uint8_t *endpoint)
#endif
{
    int i, rc;
    char last_open_dev_err[OPEN_DEV_ERROR_MESSAGE_LENGTH];
    double elapsedTime = 0;
    highres_time_t t1, t2;

    if (device_address == NULL) {
        return USB_BOOT_ERROR;
    }

    usleep(100000);
    if(usb_loglevel > 1)
    {
        if(timeout == -1)
            fprintf(stderr, "Starting wait for connect, no timeout\n");
        else if(timeout == 0)
            fprintf(stderr, "Trying to connect\n");
        else fprintf(stderr, "Starting wait for connect with %ums timeout\n", timeout);
    }
    last_open_dev_err[0] = 0;
    i = 0;
    for(;;)
    {
        highres_gettime(&t1);
        int addr_size = strlen(device_address);
#if (!defined(_WIN32) && !defined(_WIN64) )
        rc = usb_find_device_with_bcd(0, (char*)device_address, addr_size, (void**)dev,
                                      DEFAULT_VID, get_pid_by_name(device_address), bcdusb);
#else
        rc = usb_find_device(0, (char *)device_address, addr_size, (void **)dev,
            DEFAULT_VID, get_pid_by_name(device_address));
#endif
        if(rc < 0)
            return USB_BOOT_ERROR;
        if(!rc)
        {
#if (!defined(_WIN32) && !defined(_WIN64) )
            *devh = usb_open_device(*dev, endpoint, last_open_dev_err, OPEN_DEV_ERROR_MESSAGE_LENGTH);
#else
            *devh = usb_open_device(*dev, endpoint, 0, last_open_dev_err, OPEN_DEV_ERROR_MESSAGE_LENGTH);
#endif
            if(*devh != NULL)
            {
                if(usb_loglevel > 1)
                    fprintf(stderr, "Found and opened device\n");
                return 0;
            }
#if (!defined(_WIN32) && !defined(_WIN64) )
            libusb_unref_device(*dev);
            *dev = NULL;
#endif
        }
        highres_gettime(&t2);
        elapsedTime += highres_elapsed_ms(&t1, &t2);

        if(timeout != -1)
        {
            if(usb_loglevel)
            {
                if(last_open_dev_err[0])
                    fprintf(stderr, "%s", last_open_dev_err);
                fprintf(stderr, "error: device not found!\n");
            }
            return rc ? USB_BOOT_DEVICE_NOT_FOUND : USB_BOOT_TIMEOUT;
        } else if (elapsedTime > (double)timeout) {
            return rc ? USB_BOOT_DEVICE_NOT_FOUND : USB_BOOT_TIMEOUT;
        }
        i++;
        usleep(100000);
    }
    return 0;
}

#if (!defined(_WIN32) && !defined(_WIN64) )
static int send_file(libusb_device_handle* h, uint8_t endpoint, const uint8_t* tx_buf, unsigned filesize,uint16_t bcdusb)
#else
static int send_file(libusb_device_handle *h, uint8_t endpoint, const uint8_t *tx_buf, unsigned filesize)
#endif
{
    const uint8_t *p;
    int rc;
    int wb, twb, wbr;
    double elapsedTime;
    highres_time_t t1, t2;
    int bulk_chunklen=DEFAULT_CHUNKSZ;
    elapsedTime = 0;
    twb = 0;
    p = tx_buf;

#if (!defined(_WIN32) && !defined(_WIN64) )
    if(bcdusb < 0x200) {
        bulk_chunklen = USB1_CHUNKSZ;
    }
#endif
    if(usb_loglevel > 1)
        fprintf(stderr, "Performing bulk write of %u bytes...\n", filesize);
    while((unsigned)twb < filesize)
    {
        highres_gettime(&t1);
        wb = filesize - twb;
        if(wb > bulk_chunklen)
            wb = bulk_chunklen;
        wbr = 0;
#if (!defined(_WIN32) && !defined(_WIN64) )
        rc = libusb_bulk_transfer(h, endpoint, (void *)p, wb, &wbr, write_timeout);
#else
        rc = usb_bulk_write(h, endpoint, (void *)p, wb, &wbr, write_timeout);
#endif
        if(rc || (wb != wbr))
        {
            if(rc == LIBUSB_ERROR_NO_DEVICE)
                break;
            if(usb_loglevel)
                fprintf(stderr, "bulk write: %s (%d bytes written, %d bytes to write)\n", libusb_strerror(rc), wbr, wb);
            if(rc == LIBUSB_ERROR_TIMEOUT)
                return USB_BOOT_TIMEOUT;
            else return USB_BOOT_ERROR;
        }
        highres_gettime(&t2);
        elapsedTime += highres_elapsed_ms(&t1, &t2);
        if (elapsedTime > DEFAULT_SEND_FILE_TIMEOUT) {
            return USB_BOOT_TIMEOUT;
        }
        twb += wbr;
        p += wbr;
    }
    if(usb_loglevel > 1)
    {
        double MBpS = ((double)filesize / 1048576.) / (elapsedTime * 0.001);
        fprintf(stderr, "Successfully sent %u bytes of data in %lf ms (%lf MB/s)\n", filesize, elapsedTime, MBpS);
    }
    return 0;
}

int usb_boot(const char *addr, const void *mvcmd, unsigned size)
{
    int rc = 0;
    uint8_t endpoint;

#if (defined(_WIN32) || defined(_WIN64) )
    void *dev = NULL;
    struct _usb_han *h;

    rc = wait_findopen(addr, connect_timeout, &dev, &h, &endpoint);
    if(rc) {
        usb_close_device(h);
        usb_free_device(dev);
        return rc;
    }
    rc = send_file(h, endpoint, mvcmd, size);
    usb_close_device(h);
    usb_free_device(dev);
#else
    libusb_device *dev;
    libusb_device_handle *h;
    uint16_t bcdusb=-1;

    rc = wait_findopen(addr, connect_timeout, &dev, &h, &endpoint,&bcdusb);

    if(rc) {
        return rc;
    }
    rc = send_file(h, endpoint, mvcmd, size,bcdusb);
    if (h) {
        libusb_release_interface(h, 0);
        libusb_close(h);
    }
    if (dev) {
        libusb_unref_device(dev);
    }

#endif
    return rc;
}
