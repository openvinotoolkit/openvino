/*
* Copyright 2017-2019 Intel Corporation.
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

// Communication protocol used
// Few functions could be called without XLink initialization
#ifdef USE_PCIE
int gl_protocol = PCIE;
#else   // use USB as default
int gl_protocol = USB_VSC;
#endif

#define MAX_EVENTS 64
#if (defined(_WIN32) || defined(_WIN64))
extern void initialize_usb_boot();
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
        rc = usb_find_device_with_bcd(0, (char *)path, size, (void **)&dev, DEFAULT_OPENVID, DEFAULT_OPENPID,&bcdusb);
#else
		rc = usb_find_device(0, (char *)path, size, (void **)&dev, DEFAULT_OPENVID, DEFAULT_OPENPID);
#endif
        if(rc == USB_BOOT_SUCCESS)
            break;
        usleep(1000);
    }
    if (rc == USB_BOOT_TIMEOUT || rc == USB_BOOT_DEVICE_NOT_FOUND) // Timeout
        return 0;
#if (defined(_WIN32) || defined(_WIN64) )
	h = usb_open_device(dev, NULL, 0, stderr);
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
    // FIXME USE_LINK_JTAG not compiled
#ifndef USE_PCIE
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
#endif  // USE_PCIE
    return 0;
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
    #if (!defined(USE_USB_VSC) && !defined(USE_PCIE))
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

int UsbLinkPlatformInit(int loglevel)
{
    usb_loglevel = loglevel;
#if (defined(_WIN32) || defined(_WIN64))
    initialize_usb_boot();
#endif
    return 0;
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

static int write_pending = 0;
static int read_pending = 0;


static int pcie_host_write(void *f,
                           void *data, int size,
                           unsigned int timeout)
{
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
}

static int pcie_host_read(void *f,
                          void *data, int size,
                          unsigned int timeout)
{
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
}

static int pcie_host_open(const char* devPathRead,
                          const char* devPathWrite,
                          void** fd )
{
    return pcie_init(devPathWrite, fd);
}

static int pcie_host_close(void *f)
{
    pcie_close(f);
    return 0;
}

/*############################### FUNCTION ARRAYS #################################*/
/*These arrays hold the write/read/open/close operation functions
specific for each communication protocol.
Add more functions if adding another protocol*/
int (*write_fcts[NMB_OF_PROTOCOLS])(void*, void*, int, unsigned int) = \
                            {USBLinkWrite, USBLinkWrite, pcie_host_write};
int (*read_fcts[NMB_OF_PROTOCOLS])(void*, void*, int, unsigned int) = \
                            {USBLinkRead, XLinkRead, pcie_host_read};
int (*open_fcts[NMB_OF_PROTOCOLS])(const char*, const char*, void**) = \
                            {UsbLinkPlatformConnect, UsbLinkPlatformConnect, pcie_host_open};
int (*close_fcts[NMB_OF_PROTOCOLS])(void*) = \
                            {USBLinkPlatformResetRemote, USBLinkPlatformResetRemote, pcie_host_close};


/*#################################################################################
###################################################################################
###################################### EXTERNAL ###################################
###################################################################################
##################################################################################*/
int XLinkPlatformConnect(const char* devPathRead, const char* devPathWrite, void** fd)
{
    return open_fcts[gl_protocol](devPathRead, devPathWrite, fd);
}

int XLinkWrite(void* fd, void* data, int size, unsigned int timeout)
{
    return write_fcts[gl_protocol](fd, data, size, timeout);
}

int XLinkRead(void* fd, void* data, int size, unsigned int timeout)
{
    return read_fcts[gl_protocol](fd, data, size, timeout);
}

int XLinkPlatformCloseRemote(void *fd)
{
    return close_fcts[gl_protocol](fd);
}

int XLinkPlatformInit(XLinkProtocol_t protocol, int loglevel)
{
    gl_protocol = protocol;
    usb_loglevel = loglevel;
#if (defined(_WIN32) || defined(_WIN64))
    initialize_usb_boot();
#endif
    return 0;
}

static int getDeviceName(int index, char* name, int nameSize , int pid)
{
    if (index < 0 ) {
        perror("Incorrect index value\n");
        return X_LINK_PLATFORM_ERROR;
    }
    switch (gl_protocol) {
        case PCIE: {
            return pcie_find_device_port(index, name, nameSize);
        }
        case IPC:
            perror("IPC not supported, switched to USB\n");
            break;
        case USB_CDC:
            perror("USB_CDC not supported, switch to USB\n");
            break;
        case USB_VSC:
            /*should have common device(temporary moved to 'default')*/
        default:
        {
            // At the moment there is no situation where you may need a non standard vid
            int vid = AUTO_PID;

#if (!defined(_WIN32) && !defined(_WIN64))
            uint16_t  bcdusb = -1;
            usbBootError_t rc = usb_find_device_with_bcd(index, name, nameSize, 0, vid, pid, &bcdusb);
#else
            usbBootError_t rc = usb_find_device(index, name, nameSize, 0, vid, pid);
#endif
            // TODO convert usb error to xLinkPlatformErrorCode
            return parseUsbBootError(rc);
        }
    }
    return X_LINK_PLATFORM_SUCCESS;
}

int XLinkPlatformGetDeviceName(int index, char* name, int nameSize)
{
    return getDeviceName(index, name, nameSize, AUTO_PID);
}

int XLinkPlatformGetDeviceNameExtended(int index, char* name, int nameSize, int pid)
{
    return getDeviceName(index, name, nameSize, pid);
}

int XLinkPlatformBootRemote(const char* deviceName, const char* binaryPath)
{
/* Don't try to boot FW if PCIe */
#ifdef USE_PCIE
    return 0;
#else
    long filesize;
    FILE *fp;
    char *tx_buf;
    char subaddr[28+2];
    int rc;

#ifndef USE_USB_VSC
    if (usbFdRead != -1){
        close(usbFdRead);
        usbFdRead = -1;
    }
    if (usbFdWrite != -1){
        close(usbFdWrite);
        usbFdWrite = -1;
    }
#endif  /*USE_USB_VSC*/

    // Load the executable
    fp = fopen(binaryPath, "rb");
    if(fp == NULL)
    {
        if(usb_loglevel)
            perror(binaryPath);
        return -7;
    }
    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);
    rewind(fp);    
    if(filesize <= 0 || !(tx_buf = (char*)malloc(filesize)))
    {
        if(usb_loglevel)
            perror("buffer");
        fclose(fp);
        return -3;
    }
    if(fread(tx_buf, 1, filesize, fp) != filesize)
    {
        if(usb_loglevel)
            perror(binaryPath);
        fclose(fp);
        free(tx_buf);
        return -7;
    }
    fclose(fp);

    // This will be the string to search for in /sys/dev/char links
    int chars_to_write = snprintf(subaddr, 28, "-%s:", deviceName);
    if(chars_to_write >= 28) {
        printf("Path to your boot util is too long for the char array here!\n");
    }
    // Boot it
    rc = usb_boot(deviceName, tx_buf, filesize);
    free(tx_buf);
    if(rc)
    {
        return rc;
    }
    if(usb_loglevel > 1)
        fprintf(stderr, "Boot successful, device address %s\n", deviceName);
    return 0;
#endif  // USE_PCIE
}
