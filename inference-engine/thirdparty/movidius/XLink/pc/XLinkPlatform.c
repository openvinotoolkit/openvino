// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/timeb.h>
#include <errno.h>
#include <assert.h>

#include "XLinkPlatform.h"
#include "usb_boot.h"
#include "pcie_host.h"

#define MVLOG_UNIT_NAME XLinkPlatform
#include "mvLog.h"
#include "mvStringUtils.h"

#if (defined(_WIN32) || defined(_WIN64))
#include "usb_winusb.h"
#include "gettime.h"
#include "win_pthread.h"
extern void initialize_usb_boot();
#else
#include <unistd.h>
#include <sys/wait.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <termios.h>
#include <libusb.h>
#endif  /*defined(_WIN32) || defined(_WIN64)*/

#ifdef USE_LINK_JTAG
#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif  /*USE_LINK_JTAG*/

#define USBLINK_ERROR_PRINT

#ifdef USBLINK_ERROR_PRINT
#define USBLINK_ERROR(...) printf(__VA_ARGS__)
#else
#define USBLINK_ERROR(...) (void)0
#endif  /*USBLINK_ERROR_PRINT*/
#ifdef USBLINKDEBUG


#define USBLINK_PRINT(...) printf(__VA_ARGS__)
#else
#define USBLINK_PRINT(...) (void)0
#endif  /*USBLINKDEBUG*/
#ifdef USBLINKWARN


#define USBLINK_WARN(...) printf(__VA_ARGS__)
#else
#define USBLINK_WARN(...) (void)0
#endif  /*USBLINKWARN*/

//// Defines
#define UNUSED __attribute__((unused))

#define USB_LINK_SOCKET_PORT 5678
#define MAX_EVENTS 64
#define USB_ENDPOINT_IN 0x81
#define USB_ENDPOINT_OUT 0x01

#ifndef USE_USB_VSC
int usbFdWrite = -1;
int usbFdRead = -1;
#endif  /*USE_USB_VSC*/

static int statuswaittimeout = 5;

#include <assert.h>

pthread_t readerThreadId;

#define MAX_EVENTS 64
#if (defined(_WIN32) || defined(_WIN64))
extern void initialize_usb_boot();
#define OPEN_DEV_ERROR_MESSAGE_LENGTH 128
#endif

static xLinkPlatformErrorCode_t parseUsbBootError(usbBootError_t rc) {
    switch (rc) {
        case USB_BOOT_SUCCESS:
            return X_LINK_PLATFORM_SUCCESS;
        case USB_BOOT_DEVICE_NOT_FOUND:
            return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
        case USB_BOOT_TIMEOUT:
            return X_LINK_PLATFORM_TIMEOUT;
        default:
            return X_LINK_PLATFORM_ERROR;
    }
}

static int usb_write(libusb_device_handle *f, const void *data, size_t size, unsigned int timeout)
{
    while(size > 0)
    {
        int bt, ss = size;
        if(ss > 1024*1024*5)
            ss = 1024*1024*5;
#if (defined(_WIN32) || defined(_WIN64) )
        int rc = usb_bulk_write(f, USB_ENDPOINT_OUT, (unsigned char *)data, ss, &bt, timeout);
#else
        int rc = libusb_bulk_transfer(f, USB_ENDPOINT_OUT, (unsigned char *)data, ss, &bt, timeout);
#endif
        if(rc)
            return rc;
        data = (char *)data + bt;
        size -= bt;
    }
    return 0;
}


static int usb_read(libusb_device_handle *f, void *data, size_t size, unsigned int timeout)
{
    while(size > 0)
    {
        int bt, ss = size;
        if(ss > 1024*1024*5)
            ss = 1024*1024*5;
#if (defined(_WIN32) || defined(_WIN64))
        int rc = usb_bulk_read(f, USB_ENDPOINT_IN, (unsigned char *)data, ss, &bt, timeout);
#else
        int rc = libusb_bulk_transfer(f, USB_ENDPOINT_IN,(unsigned char *)data, ss, &bt, timeout);
#endif
        if(rc)
            return rc;
        data = ((char *)data) + bt;
        size -= bt;
    }
    return 0;
}


static double seconds()
{
    static double s;
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    if(!s)
        s = ts.tv_sec + ts.tv_nsec * 1e-9;
    return ts.tv_sec + ts.tv_nsec * 1e-9 - s;
}

libusb_device_handle *usblink_open(const char *path)
{
    if (path == NULL) {
        return 0;
    }

    usbBootError_t rc = USB_BOOT_DEVICE_NOT_FOUND;
    libusb_device_handle *h = NULL;
    libusb_device *dev = NULL;
    double waittm = seconds() + statuswaittimeout;
    while(seconds() < waittm){
        int size = strlen(path);

#if (!defined(_WIN32) && !defined(_WIN64))
        uint16_t  bcdusb = -1;
        rc = usb_find_device_with_bcd(0, (char *)path, size, (void **)&dev, DEFAULT_OPENVID, DEFAULT_OPENPID, &bcdusb, 0);
#else
		rc = usb_find_device(0, (char *)path, size, (void **)&dev, DEFAULT_OPENVID, DEFAULT_OPENPID, 0);
#endif
        if(rc == USB_BOOT_SUCCESS)
            break;
        usleep(1000);
    }
    if (rc == USB_BOOT_TIMEOUT || rc == USB_BOOT_DEVICE_NOT_FOUND) // Timeout
        return 0;
#if (defined(_WIN32) || defined(_WIN64) )
	h = usb_open_device(dev, NULL, 0, stderr, OPEN_DEV_ERROR_MESSAGE_LENGTH);
	int libusb_rc = ((h != NULL) ? (0) : (-1));
	if (libusb_rc < 0)
	{
		usb_close_device(h);
		usb_free_device(dev);
		return 0;
	}
	usb_free_device(dev);
#else
    int libusb_rc = libusb_open(dev, &h);
    if (libusb_rc < 0)
    {
        libusb_unref_device(dev);
        return 0;
    }
    libusb_unref_device(dev);
    libusb_detach_kernel_driver(h, 0);
    libusb_rc = libusb_claim_interface(h, 0);
    if(libusb_rc < 0)
    {
        libusb_close(h);
        return 0;
    }
#endif
    return h;
}

void usblink_close(libusb_device_handle *f)
{
#if (defined(_WIN32) || defined(_WIN64))
    usb_close_device(f);
#else
    libusb_release_interface(f, 0);
    libusb_close(f);
#endif
}

int USBLinkWrite(void* fd, void* data, int size, unsigned int timeout)
{
    int rc = 0;
#ifndef USE_USB_VSC
    int byteCount = 0;
#ifdef USE_LINK_JTAG
    while (byteCount < size){
        byteCount += write(usbFdWrite, &((char*)data)[byteCount], size - byteCount);
        printf("write %d %d\n", byteCount, size);
    }
#else
    if(usbFdWrite < 0)
    {
        return -1;
    }
    while(byteCount < size)
    {
       int toWrite = (PACKET_LENGTH && (size - byteCount > PACKET_LENGTH)) \
                        ? PACKET_LENGTH:size - byteCount;
       int wc = write(usbFdWrite, ((char*)data) + byteCount, toWrite);

       if ( wc != toWrite)
       {
           return -2;
       }

       byteCount += toWrite;
       unsigned char acknowledge;
       int rc;
       rc = read(usbFdWrite, &acknowledge, sizeof(acknowledge));

       if ( rc < 0)
       {
           return -2;
       }

       if (acknowledge != 0xEF)
       {
           return -2;
       }
    }
#endif  /*USE_LINK_JTAG*/
#else
    rc = usb_write((libusb_device_handle *) fd, data, size, timeout);
#endif  /*USE_USB_VSC*/
    return rc;
}

 int USBLinkRead(void* fd, void* data, int size, unsigned int timeout)
{
    int rc = 0;
#ifndef USE_USB_VSC
    int nread =  0;
#ifdef USE_LINK_JTAG
    while (nread < size){
        nread += read(usbFdWrite, &((char*)data)[nread], size - nread);
        printf("read %d %d\n", nread, size);
    }
#else
    if(usbFdRead < 0)
    {
        return -1;
    }

    while(nread < size)
    {
        int toRead = (PACKET_LENGTH && (size - nread > PACKET_LENGTH)) \
                        ? PACKET_LENGTH : size - nread;

        while(toRead > 0)
        {
            rc = read(usbFdRead, &((char*)gl_protocoldata)[nread], toRead);
            if ( rc < 0)
            {
                return -2;
            }
            toRead -=rc;
            nread += rc;
        }
        unsigned char acknowledge = 0xEF;
        int wc = write(usbFdRead, &acknowledge, sizeof(acknowledge));
        if (wc != sizeof(acknowledge))
        {
            return -2;
        }
    }
#endif  /*USE_LINK_JTAG*/
#else
    rc = usb_read((libusb_device_handle *) fd, data, size, timeout);
#endif  /*USE_USB_VSC*/
    return rc;
}

int USBLinkPlatformResetRemote(void* fd)
{

#ifndef USE_USB_VSC
#ifdef USE_LINK_JTAG
    /*Nothing*/
#else
    if (usbFdRead != -1){
        close(usbFdRead);
        usbFdRead = -1;
    }
    if (usbFdWrite != -1){
        close(usbFdWrite);
        usbFdWrite = -1;
    }
#endif  /*USE_LINK_JTAG*/
#else
    usblink_close((libusb_device_handle *) fd);
#endif  /*USE_USB_VSC*/
    return -1;
}

int UsbLinkPlatformConnect(const char* devPathRead, const char* devPathWrite, void** fd)
{
    #if (!defined(USE_USB_VSC))
#ifdef USE_LINK_JTAG
    struct sockaddr_in serv_addr;
    usbFdWrite = socket(AF_INET, SOCK_STREAM, 0);
    usbFdRead = socket(AF_INET, SOCK_STREAM, 0);
    assert(usbFdWrite >=0);
    assert(usbFdRead >=0);
    memset(&serv_addr, '0', sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serv_addr.sin_port = htons(USB_LINK_SOCKET_PORT);

    if (connect(usbFdWrite, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        perror("ERROR connecting");
        exit(1);
    }
    printf("this is working\n");
    return 0;

#else
    usbFdRead= open(devPathRead, O_RDWR);
    if(usbFdRead < 0)
    {
        return -1;
    }
    // set tty to raw mode
    struct termios  tty;
    speed_t     spd;
    int rc;
    rc = tcgetattr(usbFdRead, &tty);
    if (rc < 0) {
        usbFdRead = -1;
        return -2;
    }

    spd = B115200;
    cfsetospeed(&tty, (speed_t)spd);
    cfsetispeed(&tty, (speed_t)spd);

    cfmakeraw(&tty);

    rc = tcsetattr(usbFdRead, TCSANOW, &tty);
    if (rc < 0) {
        usbFdRead = -1;
        return -2;
    }

    usbFdWrite= open(devPathWrite, O_RDWR);
    if(usbFdWrite < 0)
    {
        usbFdWrite = -1;
        return -2;
    }
    // set tty to raw mode
    rc = tcgetattr(usbFdWrite, &tty);
    if (rc < 0) {
        usbFdWrite = -1;
        return -2;
    }

    spd = B115200;
    cfsetospeed(&tty, (speed_t)spd);
    cfsetispeed(&tty, (speed_t)spd);

    cfmakeraw(&tty);

    rc = tcsetattr(usbFdWrite, TCSANOW, &tty);
    if (rc < 0) {
        usbFdWrite = -1;
        return -2;
    }
    return 0;
#endif  /*USE_LINK_JTAG*/
#else
    *fd = usblink_open(devPathWrite);
    if (*fd == 0)
    {
       /* could fail due to port name change */
       return -1;
    }

    if(*fd)
        return 0;
    else
        return -1;
#endif  /*USE_USB_VSC*/
}

void deallocateData(void* ptr,uint32_t size, uint32_t alignment)
{
    if (!ptr)
        return;
#if (defined(_WIN32) || defined(_WIN64) )
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void* allocateData(uint32_t size, uint32_t alignment)
{
    void* ret = NULL;
#if (defined(_WIN32) || defined(_WIN64) )
    ret = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ret, alignment, size) != 0) {
        perror("memalign failed");
    }
#endif
    return ret;
}


/*#################################################################################
################################### PCIe FUNCTIONS ################################
##################################################################################*/

#if (defined(_WIN32) || defined(_WIN64))
static int write_pending = 0;
static int read_pending = 0;
#endif

static int pcie_host_write(void *f,
                           void *data, int size,
                           unsigned int timeout)
{
#if (defined(_WIN32) || defined(_WIN64))
#define CHUNK_SIZE_BYTES (5ULL * 1024ULL * 1024ULL)

    while (size)
    {
        write_pending = 1;

        size_t chunk = size < CHUNK_SIZE_BYTES ? size : CHUNK_SIZE_BYTES;
        int num_written = pcie_write(f, data, chunk, timeout);

        write_pending = 0;

        if (num_written == -EAGAIN)  {
            // Let read commands be submitted
            if (read_pending > 0) {
                usleep(1000);
            }
            continue;
        }

        if (num_written < 0) {
            return num_written;
        }

        data = ((char*) data) + num_written;
        /**
         * num_written is always not greater than size
         */
        size -= num_written;
    }

    return 0;
#undef CHUNK_SIZE_BYTES
#else       // Linux case
    int left = size;

    while (left > 0)
    {
        int bt = pcie_write(f, data, left, timeout);
        if (bt < 0)
            return bt;

        data = ((char *)data) + bt;
        left -= bt;
    }

    return 0;
#endif
}

static int pcie_host_read(void *f,
                          void *data, int size,
                          unsigned int timeout)
{
#if (defined(_WIN32) || defined(_WIN64))
    while (size)
    {
        read_pending = 1;

        int num_read = pcie_read(f, data, size, timeout);

        read_pending = 0;

        if (num_read == -EAGAIN)  {
            // Let write commands be submitted
            if (write_pending > 0) {
                usleep(1000);
            }
            continue;
        }

        if(num_read < 0) {
            return num_read;
        }

        data = ((char *)data) + num_read;
        /**
         * num_read is always not greater than size
         */
        size -= num_read;
    }

    return 0;
#else       // Linux
    int left = size;

    while (left > 0)
    {
        int bt = pcie_read(f, data, left, timeout);
        if (bt < 0)
            return bt;

        data = ((char *)data) + bt;
        left -= bt;
    }

    return 0;
#endif
}

static int pcie_host_open(UNUSED const char* devPathRead,
                          const char* devPathWrite,
                          void** fd )
{
    return pcie_init(devPathWrite, fd);
}

static int pcie_host_close(void *f)
{
#if (!defined(_WIN32) && !defined(_WIN64))
    /**  For PCIe device reset is called on host side  */
    pcie_reset_device(*(int*)f);
#endif

    pcie_close(f);
    return 0;
}

/*############################### FUNCTION ARRAYS #################################*/
/*These arrays hold the write/read/open/close operation functions
specific for each communication protocol.
Add more functions if adding another protocol*/
int (*write_fcts[X_LINK_NMB_OF_PROTOCOLS])(void*, void*, int, unsigned int) = \
                            {USBLinkWrite, USBLinkWrite, pcie_host_write};
int (*read_fcts[X_LINK_NMB_OF_PROTOCOLS])(void*, void*, int, unsigned int) = \
                            {USBLinkRead, USBLinkRead, pcie_host_read};
int (*open_fcts[X_LINK_NMB_OF_PROTOCOLS])(const char*, const char*, void**) = \
                            {UsbLinkPlatformConnect, UsbLinkPlatformConnect, pcie_host_open};
int (*close_fcts[X_LINK_NMB_OF_PROTOCOLS])(void*) = \
                            {USBLinkPlatformResetRemote, USBLinkPlatformResetRemote, pcie_host_close};


/*#################################################################################
###################################################################################
###################################### EXTERNAL ###################################
###################################################################################
##################################################################################*/
int XLinkPlatformConnect(const char* devPathRead, const char* devPathWrite, XLinkProtocol_t protocol, void** fd)
{
    return open_fcts[protocol](devPathRead, devPathWrite, fd);
}

int XLinkWrite(xLinkDeviceHandle_t* deviceHandle, void* data, int size, unsigned int timeout)
{
    return write_fcts[deviceHandle->protocol](deviceHandle->xLinkFD, data, size, timeout);
}

int XLinkRead(xLinkDeviceHandle_t* deviceHandle, void* data, int size, unsigned int timeout)
{
    return read_fcts[deviceHandle->protocol](deviceHandle->xLinkFD, data, size, timeout);
}

int XLinkPlatformCloseRemote(xLinkDeviceHandle_t* deviceHandle)
{
    if(deviceHandle->protocol == X_LINK_ANY_PROTOCOL ||
        deviceHandle->protocol == X_LINK_NMB_OF_PROTOCOLS) {
        perror("No method for closing handler with protocol value equals to X_LINK_ANY_PROTOCOL and X_LINK_NMB_OF_PROTOCOLS\n");
        return X_LINK_PLATFORM_ERROR;
    }

    return close_fcts[deviceHandle->protocol](deviceHandle->xLinkFD);
}

void XLinkPlatformInit()
{
#if (defined(_WIN32) || defined(_WIN64))
    initialize_usb_boot();
#endif
}

static int getDeviceName(int index, XLinkDeviceState_t state, deviceDesc_t* out_deviceDesc,
    XLinkProtocol_t protocol, XLinkPlatform_t platform, int searchByName)
{
    if (index < 0) {
        perror("Incorrect index value\n");
        return X_LINK_PLATFORM_ERROR;
    }

    if(protocol == X_LINK_ANY_PROTOCOL ||
       protocol == X_LINK_USB_VSC) {
        // At the moment there is no situation where you may need a non standard vid
        int vid = AUTO_VID;

        int pid = AUTO_PID;
        if(state == X_LINK_UNBOOTED) {
            pid = XLinkPlatformToPid(platform);
            if(searchByName) {
                pid = get_pid_by_name(out_deviceDesc->name);
            }
        } else if(state == X_LINK_BOOTED) {
            pid = DEFAULT_OPENPID;
        }

#if (!defined(_WIN32) && !defined(_WIN64))
        uint16_t  bcdusb = -1;
        usbBootError_t rc = usb_find_device_with_bcd(index, out_deviceDesc->name, XLINK_MAX_NAME_SIZE, 0, vid, pid, &bcdusb, searchByName);
#else
        usbBootError_t rc = usb_find_device(index, out_deviceDesc->name, XLINK_MAX_NAME_SIZE, 0, vid, pid, searchByName);
#endif

        xLinkPlatformErrorCode_t xLinkRc = parseUsbBootError(rc);
        if(xLinkRc == X_LINK_PLATFORM_SUCCESS)
        {
            out_deviceDesc->protocol = X_LINK_USB_VSC;
            out_deviceDesc->platform = XLinkPlatformPidToPlatform(
                                               get_pid_by_name(out_deviceDesc->name));
            return xLinkRc;
        }
    }

    if((protocol == X_LINK_ANY_PROTOCOL ||
        protocol == X_LINK_PCIE) && !searchByName) {
        out_deviceDesc->protocol = X_LINK_PCIE;
        out_deviceDesc->platform = platform;

        // #-18686
        return pcie_find_device_port(0, out_deviceDesc->name, XLINK_MAX_NAME_SIZE);
    }

    memset(out_deviceDesc, 0, sizeof(deviceDesc_t));
    return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
}

int XLinkPlatformFindDeviceName(int index,
    XLinkDeviceState_t state,
    deviceDesc_t* in_deviceRequirements,
    deviceDesc_t* out_foundDevice)
{
    if(strnlen(in_deviceRequirements->name, XLINK_MAX_NAME_SIZE)) {
        mv_strcpy(out_foundDevice->name, XLINK_MAX_NAME_SIZE, in_deviceRequirements->name);

        return getDeviceName(index, state, out_foundDevice, in_deviceRequirements->protocol, in_deviceRequirements->platform, 1);
    }

    memset(out_foundDevice->name, 0, XLINK_MAX_NAME_SIZE);
    return getDeviceName(index, state, out_foundDevice, in_deviceRequirements->protocol, in_deviceRequirements->platform, 0);
}

int XLinkPlatformIsDescriptionValid(deviceDesc_t *in_deviceDesc) {
    if(!in_deviceDesc){
        return 0;
    }

    if(!strnlen(in_deviceDesc->name, XLINK_MAX_NAME_SIZE)) {
        return 1;
    }

    if(in_deviceDesc->platform != X_LINK_ANY_PLATFORM) {
        int namePid = get_pid_by_name(in_deviceDesc->name);
        int platformPid = XLinkPlatformToPid(in_deviceDesc->platform);

        return namePid == platformPid;
    }

    return 1;
}

int XLinkPlatformToPid(const XLinkPlatform_t platform) {
    switch (platform) {
        case X_LINK_MYRIAD_2: return DEFAULT_UNBOOTPID_2150;
        case X_LINK_MYRIAD_X: return DEFAULT_UNBOOTPID_2485;
        default:       return AUTO_UNBOOTED_PID;
    }
}

XLinkPlatform_t XLinkPlatformPidToPlatform(const int pid) {
    switch (pid) {
        case DEFAULT_UNBOOTPID_2150: return X_LINK_MYRIAD_2;
        case DEFAULT_UNBOOTPID_2485: return X_LINK_MYRIAD_X;
        default:       return X_LINK_ANY_PLATFORM;
    }
}

int XLinkPlatformBootRemote(deviceDesc_t* deviceDesc, const char* binaryPath)
{
    int rc = 0;
    FILE *file;
    long file_size;

    void *image_buffer;

    /* Open the mvcmd file */
    file = fopen(binaryPath, "rb");

    if(file == NULL) {
        if(usb_loglevel)
            perror(binaryPath);
        return -7;
    }

    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);
    if(file_size <= 0 || !(image_buffer = (char*)malloc(file_size)))
    {
        if(usb_loglevel)
            perror("buffer");
        fclose(file);
        return -3;
    }
    if(fread(image_buffer, 1, file_size, file) != file_size)
    {
        if(usb_loglevel)
            perror(binaryPath);
        fclose(file);
        free(image_buffer);
        return -7;
    }
    fclose(file);

    if (deviceDesc->protocol == X_LINK_PCIE) {
        // FIXME Temporary open fd to boot device and then close it. But it can cause some problem
        int* pcieFd = NULL;
        rc = pcie_init(deviceDesc->name, (void**)&pcieFd);
        if (rc) {
            free(image_buffer);
            return rc;
        }
        rc = pcie_boot_device(*(int*)pcieFd, image_buffer, file_size);
        free(image_buffer);

        pcie_close(pcieFd); // Will not check result for now
        return rc;
    } else if (deviceDesc->protocol == X_LINK_USB_VSC) {

        char subaddr[28+2];
        // This will be the string to search for in /sys/dev/char links
        int chars_to_write = snprintf(subaddr, 28, "-%s:", deviceDesc->name);
        if(chars_to_write >= 28) {
            printf("Path to your boot util is too long for the char array here!\n");
        }
        // Boot it
        rc = usb_boot(deviceDesc->name, image_buffer, file_size);
        free(image_buffer);

        if(!rc && usb_loglevel > 1) {
            fprintf(stderr, "Boot successful, device address %s\n", deviceDesc->name);
        }
        return rc;
    } else {
        printf("Selected protocol not supported\n");
        free(image_buffer);
        return -1;
    }
}
