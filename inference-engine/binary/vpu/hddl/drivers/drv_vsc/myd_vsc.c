/*
 * USB VSC driver
 */


// 1: Includes
// ----------------------------------------------------------------------------
#include <linux/kernel.h>
#include <linux/errno.h>
#include <linux/slab.h>
#include <linux/module.h>
#include <linux/kref.h>
#include <linux/uaccess.h>
#include <linux/usb.h>
#include <linux/mutex.h>
#include <linux/llist.h>
#include "myd_vsc.h"

// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------

// Define these values to match your devices
#define USB_VSC_VENDOR_ID  0x03E7
#define USB_VSC_PRODUCT_ID 0xF63B

//#define MYD_DBG(...)    pr_info(__VA_ARGS__)
#ifndef MYD_DBG
#define MYD_DBG(...)
#endif

//Get a minor range for your devices from the usb maintainer
#define USB_VSC_MINOR_BASE 192

// MAX_TRANSFER is chosen so that the VM is not stressed by
// allocations > PAGE_SIZE and the number of packets in a page
// is an integer 512 is the largest possible packet on EHCI
#define MAX_TRANSFER        (PAGE_SIZE)

//arbitrarily chosen
#define WRITES_IN_FLIGHT    8

#define to_myd_vsc_dev(d) container_of(d, struct usb_myd_vsc, kref)


// 3: Global Data (Only if absolutely necessary)
// ----------------------------------------------------------------------------

// 4: Static Local Data
// ----------------------------------------------------------------------------

//table of devices that work with this driver
static const struct usb_device_id myd_vsc_table[] = {
    { USB_DEVICE(USB_VSC_VENDOR_ID, USB_VSC_PRODUCT_ID) },
    { }                 /* Terminating entry */
};
MODULE_DEVICE_TABLE(usb, myd_vsc_table);


//structure to keep write struct urb
struct myd_urb {
    struct urb urb;
    struct llist_node node;
};

//Structure to hold all of our device specific stuff
struct usb_myd_vsc {
    struct usb_device   *udev;          /* the usb device for this device */
    struct usb_interface    *interface;     /* the interface for this device */
    struct semaphore    limit_sem;      /* limiting the number of writes in progress */
    struct usb_anchor   submitted;      /* in case we need to retract our submissions */
    struct urb      *bulk_in_urb;       /* the urb to read data with */
    unsigned char           *bulk_in_buffer;    /* the buffer to receive data */
    size_t          bulk_in_size;       /* the size of the receive buffer */
    size_t          bulk_in_filled;     /* number of bytes in the buffer */
    size_t          bulk_in_copied;     /* already copied to user space */
    __u8            bulk_in_endpointAddr;   /* the address of the bulk in endpoint */
    __u8            bulk_out_endpointAddr;  /* the address of the bulk out endpoint */
    size_t          bulk_out_maxpacketsize; /* the max packet size of the bulk out endpoint */
    struct myd_urb      bulk_out_urbs[WRITES_IN_FLIGHT]; /* the urb to read data with */
    struct llist_head   bulk_out_urb_head;  /* llist head for free
                               struct urb */
    spinlock_t      urb_lock;       /* lock for bulk out urb */
    int         errors;         /* the last request tanked */
    bool            ongoing_read;       /* a read is going on */
    spinlock_t      err_lock;       /* lock for errors */
    struct kref     kref;
    
    #ifndef BLOCK_DRV
    struct mutex        io_read_mutex;
    struct mutex        io_write_mutex;
    #else
    struct mutex        io_mutex;       /* synchronize I/O with disconnect */
    #endif
    wait_queue_head_t   bulk_in_wait;       /* to wait for an ongoing read */
};

static struct usb_driver myd_vsc_driver;

// 5: Static Function Prototypes
// ----------------------------------------------------------------------------

void print_hex(unsigned char* buf, int num)
{
        int i;
        i = 0;
        while (i<num) {
                printk("%02x", *(buf+i));
                if ((i+1)%16 == 0) printk("\n");
                else {
                     if (i&1) printk(" ");
                }
                i++;
        }
        if (i%16) printk("\n");
}


static void myd_vsc_draw_down(struct usb_myd_vsc *dev);
//static ssize_t myd_zero_cpy_read(struct usb_myd_vsc *dev, unsigned long virt,
                 //unsigned long phy, size_t count);

static int __myd_flush(struct file *file)
{
    struct usb_myd_vsc *dev;
    int res;


    dev = file->private_data;
    if (dev == NULL)
        return -ENODEV;

    /* wait for io to stop */
    #ifndef BLOCK_DRV
    mutex_lock(&dev->io_write_mutex);
    //mutex_lock(&dev->io_read_mutex);
    #else
    mutex_lock(&dev->io_mutex);
    #endif
    

    myd_vsc_draw_down(dev);

    /* read out errors, leave subsequent opens a clean slate */
    spin_lock_irq(&dev->err_lock);
    res = dev->errors ? (dev->errors == -EPIPE ? -EPIPE : -EIO) : 0;
    dev->errors = 0;
    spin_unlock_irq(&dev->err_lock);

    
    #ifndef BLOCK_DRV
    //mutex_unlock(&dev->io_read_mutex);
    mutex_unlock(&dev->io_write_mutex);
    #else
    mutex_unlock(&dev->io_mutex);
    #endif
    
    return res;
}

static void __myd_vsc_write_bulk_callback(struct urb *urb, int dev_urb)
{
    struct usb_myd_vsc *dev;
    struct myd_urb *myd_urb = NULL;

    dev = urb->context;

     
    /* sync/async unlink faults aren't errors */
    if (urb->status) {
        if (!(urb->status == -ENOENT ||
            urb->status == -ECONNRESET ||
            urb->status == -ESHUTDOWN))
            dev_err(&dev->interface->dev,
                "%s - nonzero write bulk status received: %d\n",
                __func__, urb->status);

        spin_lock(&dev->err_lock);
        dev->errors = urb->status;
        spin_unlock(&dev->err_lock);
    }

    /* free up our allocated buffer */
    #if 0
    if (urb->transfer_buffer_length < MAX_TRANSFER) {
        print_hex((unsigned char*)(urb->transfer_buffer +
              urb->transfer_buffer_length - 16), 16);
        MYD_DBG("%s size=%u\n", __func__, urb->transfer_buffer_length);
    }
    #endif

    if (dev_urb) {
        /* Add urb back to bulk_out_urb_head */
        myd_urb = container_of(urb, struct myd_urb, urb);
        llist_add(&myd_urb->node, &dev->bulk_out_urb_head);
    }
    up(&dev->limit_sem);
}

static void myd_vsc_write_bulk_zero_cpy_callback(struct urb *urb)
{

    __myd_vsc_write_bulk_callback(urb, 0);

}

static void myd_vsc_write_bulk_callback(struct urb *urb)
{

    __myd_vsc_write_bulk_callback(urb, 1);

}

static void __myd_vsc_read_bulk_callback(struct urb *urb, int dev_urb)
{
    struct usb_myd_vsc *dev;

    dev = urb->context;

    spin_lock(&dev->err_lock);
    /* sync/async unlink faults aren't errors */
    if (urb->status) {
        if (!(urb->status == -ENOENT ||
            urb->status == -ECONNRESET ||
            urb->status == -ESHUTDOWN))
            dev_err(&dev->interface->dev,
                "%s - nonzero read bulk status received: %d\n",
                __func__, urb->status);

        dev->errors = urb->status;
    } else {
        dev->bulk_in_filled = urb->actual_length;

    }
    dev->ongoing_read = 0;
    
    // Restore it for normal read
    if(dev_urb == 0){
        dev->bulk_in_urb->transfer_dma = 0;
        dev->bulk_in_urb->transfer_flags &= (~URB_NO_TRANSFER_DMA_MAP);
    }
    spin_unlock(&dev->err_lock);
    wake_up_interruptible(&dev->bulk_in_wait);
     
}

static void myd_vsc_read_bulk_zero_cpy_callback(struct urb *urb)
{

    __myd_vsc_read_bulk_callback(urb, 0);

}

static void myd_vsc_read_bulk_callback(struct urb *urb)
{

    __myd_vsc_read_bulk_callback(urb, 1);

}

static int myd_vsc_do_zero_cpy_read_io(struct usb_myd_vsc *dev, 
                                       unsigned long virt,
                                       unsigned long phy,
                                       size_t count)
{
    int rv;
    struct urb *urb = dev->bulk_in_urb;


    /* prepare a read */
    usb_fill_bulk_urb(urb,
                      dev->udev,
                      usb_rcvbulkpipe(dev->udev, dev->bulk_in_endpointAddr),
                      (void *)virt,
                      count,
                      myd_vsc_read_bulk_zero_cpy_callback,
                      (void*)dev);
    urb->transfer_dma = phy;
    urb->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;
    /* tell everybody to leave the URB alone */
    spin_lock_irq(&dev->err_lock);
    dev->ongoing_read = 1;
    spin_unlock_irq(&dev->err_lock);

    /* submit bulk in urb, which means no data to deliver */
    dev->bulk_in_filled = 0;
    dev->bulk_in_copied = 0;

    /* do it */
    rv = usb_submit_urb(dev->bulk_in_urb, GFP_KERNEL);
    if (rv < 0)
    {
        dev_err(&dev->interface->dev,
                "%s - failed submitting read urb, error %d\n",
                __func__, rv);
        rv = (rv == -ENOMEM) ? rv : -EIO;
        spin_lock_irq(&dev->err_lock);
        dev->ongoing_read = 0;
        spin_unlock_irq(&dev->err_lock);
    }

    return rv;
}

static int myd_vsc_do_read_io(struct usb_myd_vsc *dev, size_t count)
{
    int rv;

    /* prepare a read */
    usb_fill_bulk_urb(dev->bulk_in_urb,
                      dev->udev,
                      usb_rcvbulkpipe(dev->udev, dev->bulk_in_endpointAddr),
                      dev->bulk_in_buffer,
                      min(dev->bulk_in_size, count),
                      myd_vsc_read_bulk_callback,
                      dev);
    /* tell everybody to leave the URB alone */
    spin_lock_irq(&dev->err_lock);
    dev->ongoing_read = 1;
    spin_unlock_irq(&dev->err_lock);

    /* submit bulk in urb, which means no data to deliver */
    dev->bulk_in_filled = 0;
    dev->bulk_in_copied = 0;

    /* do it */
 
    rv = usb_submit_urb(dev->bulk_in_urb, GFP_KERNEL);
    if (rv < 0)
    {
        dev_err(&dev->interface->dev,
                "%s - failed submitting read urb, error %d\n",
                __func__, rv);
        rv = (rv == -ENOMEM) ? rv : -EIO;
        spin_lock_irq(&dev->err_lock);
        dev->ongoing_read = 0;
        spin_unlock_irq(&dev->err_lock);
    }

    return rv;
}


// 6: Functions Implementation
// ----------------------------------------------------------------------------

static ssize_t myd_zero_cpy_write(struct usb_myd_vsc *dev, unsigned long virt,
                  unsigned long phy, size_t size)
{
    int retval = 0;
    struct urb *urb = NULL;


    /* verify that we actually have some data to write */
    if (size == 0)
        goto exit;

    if (down_interruptible(&dev->limit_sem)) {
        retval = -ERESTARTSYS;
        goto exit;
    }

    spin_lock_irq(&dev->err_lock);
    retval = dev->errors;
    if (retval < 0) {
        /* any error is reported once */
        dev->errors = 0;
        /* to preserve notifications about reset */
        retval = (retval == -EPIPE) ? retval : -EIO;
    }
    spin_unlock_irq(&dev->err_lock);
    if (retval < 0)
        goto error;

    /* create a urb, and a buffer for it, and copy the data to the urb */
    urb = usb_alloc_urb(0, GFP_KERNEL);
    if (!urb) {
        retval = -ENOMEM;
        goto error;
    }

    /* this lock makes sure we don't submit URBs to gone devices */
    #ifndef BLOCK_DRV
    mutex_lock(&dev->io_write_mutex);
    #else
    mutex_lock(&dev->io_mutex);
    #endif

    if (!dev->interface) {      /* disconnect() was called */
        #ifndef BLOCK_DRV  
        mutex_unlock(&dev->io_write_mutex);
        #else
        mutex_unlock(&dev->io_mutex);
        #endif


        retval = -ENODEV;
        goto error;
    }

    urb->transfer_dma = phy;
  
     

    /* initialize the urb properly */
    usb_fill_bulk_urb(urb, dev->udev,
        usb_sndbulkpipe(dev->udev, dev->bulk_out_endpointAddr),
        (void*)virt, size, myd_vsc_write_bulk_zero_cpy_callback, dev);
    urb->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;
    usb_anchor_urb(urb, &dev->submitted);

    /* send the data out the bulk port */
    retval = usb_submit_urb(urb, GFP_KERNEL);
    
    #ifndef BLOCK_DRV  
    mutex_unlock(&dev->io_write_mutex);
    #else
    mutex_unlock(&dev->io_mutex);
    #endif


    if (retval) {
        dev_err(&dev->interface->dev,
            "%s - failed submitting write urb, error %d\n",
            __func__, retval);
        goto error_unanchor;
    }

    /*
     * release our reference to this urb, the USB core will eventually free
     * it entirely
     */
    usb_free_urb(urb);

    return size;

    error_unanchor:
    usb_unanchor_urb(urb);
    error:
    if (urb) {
        usb_free_urb(urb);
    }
    up(&dev->limit_sem);

    exit:
    
    return retval;
}

static ssize_t myd_vsc_write(struct file *file, const char *user_buffer,
              size_t count, loff_t *ppos)
{
    struct usb_myd_vsc *dev;
    int retval = 0;
    struct urb *urb = NULL;
    size_t writesize = min(count, (size_t)MAX_TRANSFER);
    int two_urbs = 0;
    size_t urbsize;
    struct myd_urb *myd_urb = NULL;
    struct llist_node *node = NULL;

    dev = file->private_data;
 
    /* verify that we actually have some data to write */
    if (count == 0)
        goto exit;

    if ((writesize < MAX_TRANSFER) &&
        (writesize > dev->bulk_out_maxpacketsize) &&
        (writesize%(dev->bulk_out_maxpacketsize) != 0)) {
        two_urbs = 1;
        urbsize = writesize - writesize%(dev->bulk_out_maxpacketsize);
    } else {
        urbsize = writesize;
    }

    TwoUrbs:  

    /*
     * limit the number of URBs in flight to stop a user from using up all
     * RAM
     */
    if (!(file->f_flags & O_NONBLOCK)) {
        if (down_interruptible(&dev->limit_sem)) {
            retval = -ERESTARTSYS;
            goto exit;
        }
    } else {
        if (down_trylock(&dev->limit_sem)) {
            retval = -EAGAIN;
            goto exit;
        }
    }

    spin_lock_irq(&dev->err_lock);
    retval = dev->errors;
    if (retval < 0) {
        /* any error is reported once */
        dev->errors = 0;
        /* to preserve notifications about reset */
        retval = (retval == -EPIPE) ? retval : -EIO;
    }
    spin_unlock_irq(&dev->err_lock);
    if (retval < 0)
        goto error;

    /* Get a urb with MAX_TRANSFER buffer, and copy the data to the urb */
    spin_lock_irq(&dev->urb_lock);
    node = llist_del_first(&dev->bulk_out_urb_head);
    spin_unlock_irq(&dev->urb_lock);
    if (NULL == node) {
        retval = -EAGAIN;
        goto error;
    }
    myd_urb = llist_entry(node, struct myd_urb, node);
    urb = &(myd_urb->urb);

    if (copy_from_user(urb->transfer_buffer, user_buffer, urbsize)) {
        retval = -EFAULT;
        goto error;
    }

    /* this lock makes sure we don't submit URBs to gone devices */
    #ifndef BLOCK_DRV
    mutex_lock(&dev->io_write_mutex);
    #else
    mutex_lock(&dev->io_mutex);
    #endif

    if (!dev->interface) {      /* disconnect() was called */
        #ifndef BLOCK_DRV
        mutex_unlock(&dev->io_write_mutex);
        #else
        mutex_unlock(&dev->io_mutex);
        #endif
        retval = -ENODEV;
        goto error;
    }

    /* initialize the urb properly */
    usb_fill_bulk_urb(urb, dev->udev,
              usb_sndbulkpipe(dev->udev, dev->bulk_out_endpointAddr),
              urb->transfer_buffer, urbsize,
              myd_vsc_write_bulk_callback, dev);
    urb->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;
    usb_anchor_urb(urb, &dev->submitted);

    /* send the data out the bulk port */
 
    retval = usb_submit_urb(urb, GFP_KERNEL);
    
    #ifndef BLOCK_DRV
    mutex_unlock(&dev->io_write_mutex);
    #else 
    mutex_unlock(&dev->io_mutex);
    #endif

    if (retval) {
        dev_err(&dev->interface->dev,
            "%s - failed submitting write urb, error %d\n",
            __func__, retval);
        goto error_unanchor;
    }
    urb = NULL;
    myd_urb = NULL;

    if (two_urbs) {

        two_urbs = 0;
        user_buffer += urbsize;
        urbsize = writesize%(dev->bulk_out_maxpacketsize);
        goto TwoUrbs;
    }

    return writesize;

    error_unanchor:
    usb_unanchor_urb(urb);

    error:
    if (myd_urb) {
        llist_add(&myd_urb->node, &dev->bulk_out_urb_head);
    }
    up(&dev->limit_sem);

    exit:

    return retval;
}



static ssize_t myd_zero_cpy_read(struct usb_myd_vsc *dev, 
                                 unsigned long virt,
                                 unsigned long phy, 
                                 size_t count,
                                 unsigned int timeout_msec)
{
    int retval = 0;
    int rv = 0;
    bool ongoing_io;
    

    /* if we cannot read at all, return EOF */
    if (!dev->bulk_in_urb || !count)
    {
        goto restore;
    }
    /* no concurrent readers */
    #ifndef BLOCK_DRV
    retval = mutex_lock_interruptible(&dev->io_read_mutex);
    #else
    retval = mutex_lock_interruptible(&dev->io_mutex);
    #endif
    if (retval < 0)
    {
        rv = retval;
        goto restore;
    }
    
    if (!dev->interface) {
        /* disconnect() was called */
        retval = -ENODEV;
        goto exit;
    }
        /* if IO is under way, we must not touch things */
    retry:
        spin_lock_irq(&dev->err_lock);
        ongoing_io = dev->ongoing_read;
        spin_unlock_irq(&dev->err_lock);
    
        if (ongoing_io) {
            /*
             * IO may take forever
             * hence wait in an interruptible state
             */
            if(timeout_msec == 0) {
                rv = wait_event_interruptible(dev->bulk_in_wait, (!dev->ongoing_read));
                if (rv < 0) {
                    goto exit;
                }
            } else {
                unsigned long timeout_j = msecs_to_jiffies(timeout_msec);
                rv = wait_event_interruptible_timeout(dev->bulk_in_wait, (!dev->ongoing_read), timeout_j);
                if (rv <= 0) {
                    if(rv == 0) rv = -ETIMEDOUT;
                    goto exit;
                }
            }

        }
    
        /* errors must be reported */
        rv = dev->errors;
        if (rv < 0) {
            /* any error is reported once */
            dev->errors = 0;
            /* to preserve notifications about reset */
            rv = (rv == -EPIPE) ? rv : -EIO;
            /* report it */
            goto exit;
        }
    
        /*
         * if the buffer is filled we may satisfy the read
         * else we need to start IO
         */
    
        if (dev->bulk_in_filled) {
            /* we had read data */
            size_t available = dev->bulk_in_filled - dev->bulk_in_copied;
                
            if (!available) {
                /*
                 * all data has been used
                 * actual IO needs to be done
                 */
                rv = myd_vsc_do_zero_cpy_read_io(dev, virt, phy,count);
                if (rv < 0)
                    goto exit;
                else
                    goto retry;
            }

            dev->bulk_in_filled = 0;
            dev->bulk_in_copied = 0;
        } else {
            /* no data in the buffer */
            rv = myd_vsc_do_zero_cpy_read_io(dev, virt, phy,count);
            if (rv < 0)
                goto exit;
            else
                goto retry;
        }
    exit:
        #ifndef BLOCK_DRV
        mutex_unlock(&dev->io_read_mutex);
        #else
        mutex_unlock(&dev->io_mutex);
        #endif

    restore:
 
        return rv;
}

static ssize_t myd_vsc_read(struct file *file, char *buffer, size_t count,
             loff_t *ppos)
{  
    struct usb_myd_vsc *dev;
    int rv;
    bool ongoing_io;


    dev = file->private_data;

    /* if we cannot read at all, return EOF */
    if (!dev->bulk_in_urb || !count)
        return 0;

    /* no concurrent readers */
    #ifndef BLOCK_DRV
    rv = mutex_lock_interruptible(&dev->io_read_mutex);
    #else
    rv = mutex_lock_interruptible(&dev->io_mutex);
    #endif
    if (rv < 0)
        return rv;


    if (!dev->interface) {      /* disconnect() was called */
        rv = -ENODEV;
        goto exit;
    }
    

    /* if IO is under way, we must not touch things */
 retry:
    spin_lock_irq(&dev->err_lock);
    ongoing_io = dev->ongoing_read;
    spin_unlock_irq(&dev->err_lock);

    if (ongoing_io) {
        /* nonblocking IO shall not wait */
        if (file->f_flags & O_NONBLOCK) {
            rv = -EAGAIN;
            goto exit;
        }
        /*
         * IO may take forever
         * hence wait in an interruptible state
         */

        rv = wait_event_interruptible(dev->bulk_in_wait, (!dev->ongoing_read));
        if (rv < 0) {
            goto exit;
        }
    }

    /* errors must be reported */
    rv = dev->errors;
    if (rv < 0) {
        /* any error is reported once */
        dev->errors = 0;
        /* to preserve notifications about reset */
        rv = (rv == -EPIPE) ? rv : -EIO;
        /* report it */
        goto exit;
    }

    /*
     * if the buffer is filled we may satisfy the read
     * else we need to start IO
     */
    if (dev->bulk_in_filled) {
        /* we had read data */
        size_t available = dev->bulk_in_filled - dev->bulk_in_copied;
        size_t chunk = min(available, count);


        if (!available) {
            /*
             * all data has been used
             * actual IO needs to be done
             */
            rv = myd_vsc_do_read_io(dev, count);
            if (rv < 0)
                goto exit;
            else
                goto retry;
        }
        /*
         * data is available
         * chunk tells us how much shall be copied
         */

        if (copy_to_user(buffer,
                         dev->bulk_in_buffer + dev->bulk_in_copied,
                         chunk))
            rv = -EFAULT;
        else
            rv = chunk;

        dev->bulk_in_filled = 0;
        dev->bulk_in_copied = 0;
    } else {
        /* no data in the buffer */
        rv = myd_vsc_do_read_io(dev, count);
        if (rv < 0)
            goto exit;
        else
            goto retry;
    }
 exit:
    #ifndef BLOCK_DRV
    mutex_unlock(&dev->io_read_mutex);
    #else
    mutex_unlock(&dev->io_mutex);
    #endif
    return rv;
}

static ssize_t myd_raw_read(struct file *file, char *buffer, size_t count, unsigned int timeout_msec)
{
    struct usb_myd_vsc *dev;
    int rv;
    bool ongoing_io;


    dev = file->private_data;
 

    /* if we cannot read at all, return EOF */
    if (!dev->bulk_in_urb || !count)
        return 0;

    /* no concurrent readers */
#ifndef BLOCK_DRV
    rv = mutex_lock_interruptible(&dev->io_read_mutex);
#else
    rv = mutex_lock_interruptible(&dev->io_mutex);
#endif
    if (rv < 0)
        return rv;


    if (!dev->interface) {      /* disconnect() was called */
        rv = -ENODEV;
        goto exit;
    }


    /* if IO is under way, we must not touch things */
    retry:
    spin_lock_irq(&dev->err_lock);
    ongoing_io = dev->ongoing_read;
    spin_unlock_irq(&dev->err_lock);

    if (ongoing_io) {
        /* nonblocking IO shall not wait */
        if (file->f_flags & O_NONBLOCK) {
            rv = -EAGAIN;
            goto exit;
        }
        /*
         * IO may take forever
         * hence wait in an interruptible state
         */
        if(timeout_msec == 0) {
            rv = wait_event_interruptible(dev->bulk_in_wait, (!dev->ongoing_read));
            if (rv < 0) {
                goto exit;
            }
        } else {
            unsigned long timeout_j = msecs_to_jiffies(timeout_msec);
            rv = wait_event_interruptible_timeout(dev->bulk_in_wait, (!dev->ongoing_read), timeout_j);
            if(rv <= 0) {
                if(rv == 0) rv = -ETIMEDOUT;
                goto exit;
            }
        }

    }

    /* errors must be reported */
    rv = dev->errors;
    if (rv < 0) {
        /* any error is reported once */
        dev->errors = 0;
        /* to preserve notifications about reset */
        rv = (rv == -EPIPE) ? rv : -EIO;
        /* report it */
        goto exit;
    }

    /*
     * if the buffer is filled we may satisfy the read
     * else we need to start IO
     */
    if (dev->bulk_in_filled) {
        /* we had read data */
        size_t available = dev->bulk_in_filled - dev->bulk_in_copied;
        size_t chunk = min(available, count);


        if (!available) {
            /*
             * all data has been used
             * actual IO needs to be done
             */
            rv = myd_vsc_do_read_io(dev, count);
            if (rv < 0)
                goto exit;
            else
                goto retry;
        }
        /*
         * data is available
         * chunk tells us how much shall be copied
         */

        if (copy_to_user(buffer,
                         dev->bulk_in_buffer + dev->bulk_in_copied,
                         chunk))
            rv = -EFAULT;
        else
            rv = chunk;

        dev->bulk_in_filled = 0;
        dev->bulk_in_copied = 0;
    } else {
        /* no data in the buffer */
        rv = myd_vsc_do_read_io(dev, count);
        if (rv < 0)
            goto exit;
        else
            goto retry;
    }
    exit:
#ifndef BLOCK_DRV
    mutex_unlock(&dev->io_read_mutex);
#else
    mutex_unlock(&dev->io_mutex);
#endif
    return rv;
}



static long myd_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    int retval = 0;
    struct usb_myd_vsc *dev;
    myd_write_data_t wdata;
    myd_read_data_t rdata;
    read_data_t read_data;

    dev = file->private_data;
    if (_IOC_TYPE(cmd) != IOC_MYD_TYPE ||
        _IOC_NR(cmd) < IOC_MYD_MIN_NR  ||
        _IOC_NR(cmd) > IOC_MYD_MAX_NR) {
        dev_err(&(dev->udev->dev), "invalid ioctl(type %d, nr %d, size %d)\n",
               _IOC_TYPE(cmd),
               _IOC_NR(cmd), _IOC_SIZE(cmd));
        return -EINVAL;
    }
    switch(cmd) {
    case IOC_MYD_WRITE:
        if (copy_from_user(&wdata, (void*)arg, sizeof(myd_write_data_t))) {
            retval = -EFAULT;
        } else {
            retval = myd_zero_cpy_write(dev, wdata.cpu_addr, wdata.phys_addr, wdata.len);
        }
        break;
        
    case IOC_MYD_READ:
        if (copy_from_user(&rdata, (void*)arg, sizeof(myd_read_data_t))) {
            retval = -EFAULT;
        } else {
            retval = myd_zero_cpy_read(dev, rdata.cpu_addr, rdata.phys_addr, rdata.len, rdata.timeout_msec);
        }
        break;
    case RAW_MYD_READ:
        if(copy_from_user(&read_data, (void*)arg, sizeof(read_data_t))) {
            retval = -EFAULT;
        } else {
            retval = myd_raw_read(file, read_data.data, read_data.ss, read_data.timeout_msec);
        }
        break;
    default:
        return -EINVAL;
    }
    return retval;
}

static void myd_vsc_delete(struct kref *kref)
{
    struct usb_myd_vsc *dev = to_myd_vsc_dev(kref);
    struct llist_node *head = llist_del_all(&dev->bulk_out_urb_head);
    struct myd_urb *myd_urb = NULL;
    struct urb* urb = NULL;

    llist_for_each_entry(myd_urb, head, node) {
        urb = &(myd_urb->urb);
        usb_free_coherent(urb->dev, urb->transfer_buffer_length,
              urb->transfer_buffer, urb->transfer_dma);
        usb_init_urb(urb);
    }

    usb_free_urb(dev->bulk_in_urb);
    usb_put_dev(dev->udev);
    kfree(dev->bulk_in_buffer);
    kfree(dev);
}

static int myd_vsc_open(struct inode *inode, struct file *file)
{ 
    struct usb_myd_vsc *dev;
    struct usb_interface *interface;
    int subminor;
    int retval = 0;


    subminor = iminor(inode);

    interface = usb_find_interface(&myd_vsc_driver, subminor);
    if (!interface) {
        pr_err("%s - error, can't find device for minor %d\n",
            __func__, subminor);
        retval = -ENODEV;
        goto exit;
    }

    dev = usb_get_intfdata(interface);
    if (!dev) {
        retval = -ENODEV;
        goto exit;
    }

    retval = usb_autopm_get_interface(interface);
    if (retval)
        goto exit;

    /* increment our usage count for the device */
    kref_get(&dev->kref);

    /* save our object in the file's private structure */
    file->private_data = dev;

 exit:
    return retval;
}

static int myd_vsc_release(struct inode *inode, struct file *file)
{
    struct usb_myd_vsc *dev;

    pr_info("%s entrance\n", __func__);
    dev = file->private_data;
    if (dev == NULL)
        return -ENODEV;

    /* allow the device to be autosuspended */
    #ifndef BLOCK_DRV
    mutex_lock(&dev->io_write_mutex);
    mutex_lock(&dev->io_read_mutex);
    #else
    mutex_lock(&dev->io_mutex);
    #endif
    


    if (dev->interface)
        usb_autopm_put_interface(dev->interface);
    
    #ifndef BLOCK_DRV
    mutex_unlock(&dev->io_read_mutex);
    mutex_unlock(&dev->io_write_mutex);
    #else
    mutex_unlock(&dev->io_mutex);
    #endif

    /* decrement the count on our device */
    kref_put(&dev->kref, myd_vsc_delete);
    return 0;
}


static int myd_vsc_flush(struct file *file, fl_owner_t id)
{

    return __myd_flush(file);
}

static int myd_vsc_fsync(struct file *file, loff_t start, loff_t end, int datasync)
{

    return __myd_flush(file);
}


static const struct file_operations myd_vsc_fops = {
    .owner =    THIS_MODULE,
    .read =     myd_vsc_read,
    .write =    myd_vsc_write,
    .open =     myd_vsc_open,
    .release =  myd_vsc_release,
    .flush =    myd_vsc_flush,
    .fsync =    myd_vsc_fsync,
    .llseek =   noop_llseek,
    .unlocked_ioctl =   myd_ioctl,
};

/*
 * usb class driver info in order to get a minor number from the usb core,
 * and to have the device registered with the driver core
 */
static struct usb_class_driver myd_vsc_class = {
    .name =     "myriad%d",
    .fops =     &myd_vsc_fops,
    .minor_base =   USB_VSC_MINOR_BASE,
};


/****************************************************** usb driver VS device ********************************************************/

//Called to see if the driver is willing to manage a particular interface on a device. 
//If it is, probe returns zero and uses usb_set_intfdata to associate driver-specific data with the interface. 
//It may also use usb_set_interface to specify the appropriate altsetting. 
//If unwilling to manage the interface, return -ENODEV, if genuine IO errors occured, an appropriate negative errno value.
static int myd_vsc_probe(struct usb_interface *interface,
              const struct usb_device_id *id)
{ 
    struct usb_myd_vsc *dev;
    struct usb_host_interface *iface_desc;
    struct usb_endpoint_descriptor *endpoint;
    size_t buffer_size = 0;
    int i;
    int urb_idx;
    int retval = -ENOMEM;
    unsigned char* buf;
    struct urb* urb = NULL;

    /* allocate memory for our device state and initialize it */
    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev) {
        dev_err(&interface->dev, "Out of memory\n");
        goto error;
    }
    kref_init(&dev->kref);
    sema_init(&dev->limit_sem, WRITES_IN_FLIGHT);
    
    #ifndef BLOCK_DRV
    mutex_init(&dev->io_read_mutex);
    mutex_init(&dev->io_write_mutex);
    #else
    mutex_init(&dev->io_mutex);
    #endif
    spin_lock_init(&dev->err_lock);
    spin_lock_init(&dev->urb_lock);
    init_llist_head(&dev->bulk_out_urb_head);
    init_usb_anchor(&dev->submitted);
    init_waitqueue_head(&dev->bulk_in_wait);

    dev->udev = usb_get_dev(interface_to_usbdev(interface));
    dev->interface = interface;

    /* set up the endpoint information */
    /* use only the first bulk-in and bulk-out endpoints */
    iface_desc = interface->cur_altsetting;
    for (i = 0; i < iface_desc->desc.bNumEndpoints; ++i) {
        endpoint = &iface_desc->endpoint[i].desc;

        if (!dev->bulk_in_endpointAddr &&
            usb_endpoint_is_bulk_in(endpoint)) {
            /* we found a bulk in endpoint */
            buffer_size = usb_endpoint_maxp(endpoint)*1024;
            dev->bulk_in_size = buffer_size;
            dev->bulk_in_endpointAddr = endpoint->bEndpointAddress;
            dev->bulk_in_buffer = kmalloc(buffer_size, GFP_KERNEL);
            if (!dev->bulk_in_buffer) {
                dev_err(&interface->dev,
                    "Could not allocate bulk_in_buffer\n");
                goto error;
            }
            dev->bulk_in_urb = usb_alloc_urb(0, GFP_KERNEL);
            if (!dev->bulk_in_urb) {
                dev_err(&interface->dev,
                    "Could not allocate bulk_in_urb\n");
                goto error;
            }
        }

        if (!dev->bulk_out_endpointAddr &&
            usb_endpoint_is_bulk_out(endpoint)) {
            /* we found a bulk out endpoint */
            dev->bulk_out_endpointAddr = endpoint->bEndpointAddress;
            dev->bulk_out_maxpacketsize =
                usb_endpoint_maxp(endpoint);

            /* Create WRITES_IN_FLIGHT urbs with MAX_TRANSFER buf for write */
            for (urb_idx=0; urb_idx < WRITES_IN_FLIGHT;
                  urb_idx++) {
                urb = &(dev->bulk_out_urbs[urb_idx].urb);
                usb_init_urb(urb);
                buf = usb_alloc_coherent(dev->udev,
                               MAX_TRANSFER,
                             GFP_KERNEL,
                             &urb->transfer_dma);
                if (!buf) {
                    retval = -ENOMEM;
                    goto error;
                }
                urb->transfer_buffer = buf;
                llist_add(&(dev->bulk_out_urbs[urb_idx].node),
                      &dev->bulk_out_urb_head);
            }

        }
    }
    if (!(dev->bulk_in_endpointAddr && dev->bulk_out_endpointAddr)) {
        dev_err(&interface->dev,
            "Could not find both bulk-in and bulk-out endpoints\n");
        goto error;
    }

    /* save our data pointer in this interface device */
    usb_set_intfdata(interface, dev);

    /* we can register the device now, as it is ready */
    retval = usb_register_dev(interface, &myd_vsc_class);
    if (retval) {
        /* something prevented us from registering this driver */
        dev_err(&interface->dev,
            "Not able to get a minor for this device.\n");
        usb_set_intfdata(interface, NULL);
        goto error;
    }

    /* let the user know what node this device is now attached to */
    dev_info(&interface->dev,
         "USB VSC device now attached to USBVSC-%d",
         interface->minor);

    return 0;

 error:
    if (dev) {
        kfree(dev->bulk_in_buffer);
        /* this frees allocated memory */
        kref_put(&dev->kref, myd_vsc_delete);
    }
    kfree(dev);
    return retval;
}


//Called when the interface is no longer accessible, usually because its device 
//has been (or is being) disconnected or the driver module is being unloaded.
static void myd_vsc_disconnect(struct usb_interface *interface)
{
    struct usb_myd_vsc *dev;
    int minor = interface->minor;

    pr_info("%s entrance\n", __func__);
    
    dev = usb_get_intfdata(interface);
    #ifndef BLOCK_DRV
    mutex_lock(&dev->io_write_mutex);
    mutex_lock(&dev->io_read_mutex);
    #else
    mutex_lock(&dev->io_mutex);
    #endif
    
    usb_set_intfdata(interface, NULL);

    /* give back our minor */
    usb_deregister_dev(interface, &myd_vsc_class);

    /* prevent more I/O from starting */
    
    dev->interface = NULL;
    
    #ifndef BLOCK_DRV
    mutex_unlock(&dev->io_read_mutex);
    mutex_unlock(&dev->io_write_mutex);
    #else
    mutex_unlock(&dev->io_mutex);
    #endif   

    usb_kill_anchored_urbs(&dev->submitted);

    /* decrement our usage count */
    kref_put(&dev->kref, myd_vsc_delete);

    dev_info(&interface->dev, "USB VSC #%d now disconnected", minor);
}


/************************************* VSC draw down ??? ************************/
static void myd_vsc_draw_down(struct usb_myd_vsc *dev)
{
    int time;

    time = usb_wait_anchor_empty_timeout(&dev->submitted, 2000);
    if (!time) {

        usb_kill_anchored_urbs(&dev->submitted);
    }
    usb_kill_urb(dev->bulk_in_urb);
}


//Called when the device is going to be suspended by the system.
static int myd_vsc_suspend(struct usb_interface *intf, pm_message_t message)
{
    struct usb_myd_vsc *dev = usb_get_intfdata(intf);


    if (!dev)
        return 0;
    myd_vsc_draw_down(dev);   /************************************* VSC draw down ??? ************************/
    return 0;
}


//Called when the device is being resumed by the system.
static int myd_vsc_resume(struct usb_interface *intf)
{

    return 0;
}


//called by usb_reset_device when the device is about to be reset.
static int myd_vsc_pre_reset(struct usb_interface *intf)
{
    struct usb_myd_vsc *dev = usb_get_intfdata(intf);

    #ifndef BLOCK_DRV
    mutex_lock(&dev->io_write_mutex);
    mutex_lock(&dev->io_read_mutex);
    #else
    mutex_lock(&dev->io_mutex);
    #endif
    
    myd_vsc_draw_down(dev);  /************************************* VSC draw down ??? ************************/

    return 0;
}

//Called by usb_reset_device after the device has been reset
static int myd_vsc_post_reset(struct usb_interface *intf)
{
    struct usb_myd_vsc *dev = usb_get_intfdata(intf);

    /* we are sure no URBs are active - no locking needed */
    dev->errors = -EPIPE;
    
    #ifndef BLOCK_DRV
    mutex_unlock(&dev->io_read_mutex);
    mutex_unlock(&dev->io_write_mutex);
    #else
    mutex_unlock(&dev->io_mutex); 
    #endif
    return 0;
}


static struct usb_driver myd_vsc_driver = {
    .name =     "myriad_usb_vsc",
    .probe =    myd_vsc_probe,
    .disconnect =   myd_vsc_disconnect,
    .suspend =  myd_vsc_suspend,
    .resume =   myd_vsc_resume,
    .pre_reset =    myd_vsc_pre_reset,
    .post_reset =   myd_vsc_post_reset,
    .id_table = myd_vsc_table,
    .supports_autosuspend = 1,
};


module_usb_driver(myd_vsc_driver);

MODULE_LICENSE("GPL");


