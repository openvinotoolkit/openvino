/*
 * drivers/gpu/ion/ion_dummy_driver.c
 *
 * Copyright (C) 2013 Linaro, Inc
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */

#include <linux/err.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/init.h>
//#include <linux/bootmem.h>
#include <linux/memblock.h>
#include <linux/sizes.h>
#include <linux/io.h>
#include <linux/module.h>
#include "ion.h"
#include "ion_priv.h"


#define  DRIVER_NAME "ion_driver"


static struct ion_device *idev;
static struct ion_heap **heaps;


enum ion_heap_ids {
	INVALID_HEAP_ID = -1,
	ION_SYSTEM_HEAP_ID = 1,
	ION_SYSTEM_CONTIG_HEAP_ID = 2,
	ION_DMA_HEAP_ID = 4,

	ION_HEAP_ID_RESERVED = 8 /** Bit reserved for ION_SECURE flag */
};
#define ION_VMALLOC_HEAP_NAME	"vmalloc"
#define ION_KMALLOC_HEAP_NAME	"kmalloc"
#define ION_DMA_HEAP_NAME	"dmaalloc"


static struct ion_platform_heap myx_heaps[] = {
		#ifndef CentOS
		{
			.id	= ION_SYSTEM_HEAP_ID,
			.type	= ION_HEAP_TYPE_SYSTEM,
			.name	= ION_VMALLOC_HEAP_NAME,
		},
		#endif
		{
			.id	= ION_SYSTEM_CONTIG_HEAP_ID,
			.type	= ION_HEAP_TYPE_SYSTEM_CONTIG,
			.name	= ION_KMALLOC_HEAP_NAME,
		},
		{
			.id	= ION_DMA_HEAP_ID,
			.type	= ION_HEAP_TYPE_DMA,
			.name	= ION_DMA_HEAP_NAME,
		},
};

static struct ion_platform_data myx_ion_pdata = {
	.nr = ARRAY_SIZE(myx_heaps),
	.heaps = myx_heaps,
};

static int __init ion_module_init(void)
{
	int i, err;

	idev = ion_device_create(NULL);
	
	heaps = kcalloc(myx_ion_pdata.nr, sizeof(struct ion_heap *),
			GFP_KERNEL);
	if (!heaps)
		return -ENOMEM;

	for (i = 0; i < myx_ion_pdata.nr; i++) {
		struct ion_platform_heap *heap_data = &myx_ion_pdata.heaps[i];
		if(heap_data->id == ION_DMA_HEAP_ID){
 
			heap_data->priv = (void*)(ion_get_device(idev));
		}
		heaps[i] = ion_heap_create(heap_data);
		if (IS_ERR_OR_NULL(heaps[i])) {
			err = PTR_ERR(heaps[i]);
			goto err;
		}

		ion_device_add_heap(idev, heaps[i]);

	}
	return 0;
err:
	for (i = 0; i < myx_ion_pdata.nr; ++i)
		ion_heap_destroy(heaps[i]);
	kfree(heaps);

	return err;
}
//device_initcall(ion_dummy_init);

static void __exit ion_module_exit(void)
{
	int i;

	ion_device_destroy(idev);

	for (i = 0; i < myx_ion_pdata.nr; i++)
		ion_heap_destroy(heaps[i]);
	kfree(heaps);

}
//__exitcall(ion_dummy_init);



module_init(ion_module_init);
module_exit(ion_module_exit);

MODULE_AUTHOR("eason");
MODULE_DESCRIPTION("ION Driver porting");
MODULE_LICENSE("GPL");
