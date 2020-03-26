#!/bin/bash

# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

SMBUS_LINES=`lspci | grep SMBus`
LOC=`echo $SMBUS_LINES | awk '{print $1}'`
if [ "$LOC" != "" ]; then
    DEVICE=`lspci -s $LOC -n | awk '{print $3}'`
    PRODUCT_ID=`sed -r 's/(.*):(.*)/\2/' <<< $DEVICE`
else
    PRODUCT_ID="a123"
fi

echo "ATTRS{device}==\"0x${PRODUCT_ID}\", GROUP=\"users\", MODE=\"0660\"
KERNEL==\"hidraw*\", ATTRS{idVendor}==\"2c42\", ATTRS{idProduct}==\"5114\", GROUP=\"users\", MODE=\"0660\"
" > /tmp/98-hddlbsl.rules

if [ "$1" != "" ]; then
    echo "Copying to $1"
    cp /tmp/98-hddlbsl.rules $1
else
    echo "No output target provided, using /tmp/98-hddlbsl.rules"
fi

