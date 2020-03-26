# libbsl
---
this project includes some tools and a library.

1. `libbsl`: A library to reset myraidx for HDDL.
2. `bsl_reset`: A tool to reset devices based on `libbsl`
3. `smb_address_tool`: A tool on Window to find SMBus address

## Build
---
### For Linux
Install dependencies
```sh
sudo apt-get install libudev-dev libjson-c-dev
```
build with make
```
cd src
make
sudo make install
```
build with cmake

```sh
mkdir build
cd build
cmake .. -DINSTALL_USB_RULES=TRUE
make -j
sudo make install
```

### For Windows

Firstly, please make sure that you ran command prompt as an `administrator` to be able installing hddl-bsl software

```sh
mkdir build
cd build
#cmake -G"Visual Studio 14 2015 Win64" ..
cmake ..
```

after that, you can use the following command to build and install
```sh
cmake --build . --target INSTALL
```
if only need to build, then  use the following command
```sh
cmake --build .
```

you also can open `hddl-bsl.sln` to open projects and build it in VS

> hddl-bsl\build\hddl-bsl.sln

### Uninstall

Use `make`
```sh
make uninstall
```

Use `CMake`
```sh
xargs rm < install_manifest.txt
```

## interfaces
---
all interfaces list here
> include/hddl-bsl.h



## bsl_reset
---
`bsl_reset` is based on libbsl
usage:
```
bsl_reset
```
or specify a reset device
```
bsl_reset -d [mcu|io|hid]
```
you also can reset a single myx:
```
bsl_reset -d [mcu|io|hid] -i device_id
```
for example
```
bsl_reset -d hid -i 224
```
`device_id` is a number like 224, 224 means board pin is `0b111`, device id is `0b0`
```
python
>>> bin(224)
'0b11100000'
```
bsl_reset supports to discard a specific myx
```sh
bsl_reset -d io -a device_id
```

##  configure
---
for most case, we do not need a configure file for hddl-bsl, hddl-bsl supports auto scan reset device by default. but for some case, you need a config file, for example, when the host has more than one valid device(or valid device address) for hddl-bsl, then you need a config file to tell hddl-bsl which one is the right one.

currently hddl-bsl supports three reset devices:
1. MCU: it is a stm32 MCU based on SMBus
2. IOExpander:  it is also based on SMBus
3. F75114: it is HID device, based on USB

the config file is named `bsl.json`, the default path:
> ${HDDL_INSTALL_DIR}/config/bsl.json

### config file for F75114
fill bsl.json with the following lines.
```
{
  "active":"hid-f75114",
  "config": {
  }
}
```
normally the above configure is enough, but for some cases, if there is a F75114 not for HDDL cards, then we need to set the path of F75114

for linux, do it like this:

```
hddl@hddl-US-120:~/hddl-bsl/src$ lsusb
Bus 006 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 005 Device 016: ID 03e7:2485
Bus 005 Device 015: ID 03e7:2485
Bus 005 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 017: ID 03e7:2485
Bus 003 Device 016: ID 03e7:2485
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub  --> this one is the parent node
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 004: ID 2c42:5114                                ---------------> this one is f75114, bus 1, device 3
Bus 001 Device 003: ID 2c42:5114                                ---------------> this one is f75114, bus 1, device 4
Bus 001 Device 002: ID 046d:c31c Logitech, Inc. Keyboard K120
Bus 001 Device 005: ID 093a:2510 Pixart Imaging, Inc. Optical Mouse
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```
```
hddl@hddl-US-120:~/hddl-bsl/src$ lsusb -t
/:  Bus 06.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 10000M
/:  Bus 05.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 480M
    |__ Port 1: Dev 15, If 0, Class=Vendor Specific Class, Driver=, 480M
    |__ Port 2: Dev 16, If 0, Class=Vendor Specific Class, Driver=, 480M
/:  Bus 04.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 10000M
/:  Bus 03.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 480M
    |__ Port 1: Dev 16, If 0, Class=Vendor Specific Class, Driver=, 480M
    |__ Port 2: Dev 17, If 0, Class=Vendor Specific Class, Driver=, 480M
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/6p, 5000M
/:  Bus 01.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/12p, 480M            --> this one is the parent node
    |__ Port 4: Dev 5, If 0, Class=Human Interface Device, Driver=usbhid, 1.5M
    |__ Port 5: Dev 2, If 1, Class=Human Interface Device, Driver=usbhid, 1.5M
    |__ Port 5: Dev 2, If 0, Class=Human Interface Device, Driver=usbhid, 1.5M
    |__ Port 8: Dev 3, If 0, Class=Human Interface Device, Driver=usbhid, 12M  ----> bus 1, device 3,this one is f75114
    |__ Port 9: Dev 4, If 0, Class=Human Interface Device, Driver=usbhid, 12M  ----> bus 1, device 4,this one is f75114
```
then config it like this:
```
{
  "active":"hid-f75114",
  "config": {
    "parent_paths": [
      "/0x1d6b:0x0002"
    ]
  }
}
```


for window, in device manager, find the HID device, check the "Device instance path", copy it to bsl.json
notes that you need use '\\' replace '\'
```
{
  "active":"hid-f75114",
  "config": {
    "parent_paths": [
      "HID\\VID_2C42&PID_5114\\7&23A6A2C2&0&0000"
    ]
  }
}
```


###  config file for MCU and IOExpander
both MCU and IOExpander are based on SMBus, which has a i2c address.we need to set the i2c address in `bsl.json`

this is a example how to config the i2c address, one address means one hddl card.

IOExpander supports more than one hddl cards, `37` and `39` are the i2c address. 
```json
{
  "active":"ioexpander",
  "config": {
    "i2c_addr":[37,39]
  }
}
```
MCU only support one hddl-card, `31` is the i2c address.
```json
{
  "active":"mcu",
  "config": {
    "i2c_addr":[31]
  }
}
```

#### how to get i2c address for MCU

1. install i2c-i801 driver

```sh
sudo modprobe i2c-i801
```

2. list all i2c devices

```sh
i2cdetect -l

$ hddl@hddl-US-E2332:~/eason/hddl-bsl/src$ i2cdetect -l
i2c-0   i2c             i915 gmbus dpc                          I2C adapter
i2c-1   i2c             i915 gmbus dpb                          I2C adapter
i2c-2   i2c             i915 gmbus dpd                          I2C adapter
i2c-3   i2c             DPDDC-C                                 I2C adapter
i2c-4   i2c             DPDDC-E                                 I2C adapter
i2c-5   smbus           SMBus I801 adapter at f040              SMBus adapter  ==>
```

we know i2c-5 is the right one, then check the i2c address

3. check i2c address

the following case is the MCU example, the address range is from 0x18 to 0x1f,  
the 0x18 is always there, so we know it is not the pcie card, 0x1f is the i2c address for pcie card.

```sh
i2cdetect -y 5

$ hddl@hddl-US-E2332:~/eason/hddl-bsl/src$ i2cdetect -y 5
      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- 08 -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- 18 -- -- -- -- -- -- 1f                   ==>
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: 30 31 -- -- 34 35 36 -- -- -- -- -- -- -- -- --
40: -- -- -- -- 44 -- -- -- -- -- -- -- -- -- -- --
50: 50 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
```

0x1f == 31, so  config the file like the following , then it is enough.
4. fill bsl.json
```json
{
  "active":"mcu",
  "config": {
    "i2c_addr":[31]
  }
}
```
#### how to get i2c address for IOExpander

1. install i2c-i801 driver

```sh
sudo modprobe i2c-i801
```

2.  list all i2c devices

```sh
i2cdetect -l

$ hddl@hddl-US-E2332:~/eason/hddl-bsl/src$ i2cdetect -l
i2c-0   i2c             i915 gmbus dpc                          I2C adapter
i2c-1   i2c             i915 gmbus dpb                          I2C adapter
i2c-2   i2c             i915 gmbus dpd                          I2C adapter
i2c-3   i2c             DPDDC-C                                 I2C adapter
i2c-4   i2c             DPDDC-E                                 I2C adapter
i2c-5   smbus           SMBus I801 adapter at f040              SMBus adapter  ==>
```

we know i2c-5 is the right one, then check the i2c address

3.  check i2c address

the following case is the IOExpander example, the address range is from 0x20 to 0x2f.  
there are two cards, so we know the i2c address is 0x23 and 0x26.

```sh
i2cdetect -y 5

$ hddl@hddl-US-E2332:~$ i2cdetect -y 5
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- 08 -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- 18 -- -- -- -- -- -- --
20: -- -- -- 23 -- -- 26 -- -- -- -- -- -- -- -- --
30: 30 31 -- -- 34 35 36 -- -- -- -- -- -- -- -- --
40: -- -- -- -- 44 -- -- -- -- -- -- -- -- -- -- --
50: 50 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 6f
```

0x23 == 35, 0x26==38, so  config the file like the following , then it is enough.

4. fill bsl.json
```json
{
  "active":"ioexpander",
  "config": {
    "i2c_addr":[35,38]
  }
}
```

## Example: how to reset IOExpander
---
install i2c-tools
```
sudo apt install i2c-tools
```
suppose 0x23 is the i2c address
use i2c-tools to reset it
```sh
sudo i2cset -y 5 0x23 0x06 0x00 b
sudo i2cset -y 5 0x23 0x02 0x00 b
sudo i2cset -y 5 0x23 0x02 0xff b
```

use bsl_reset to reset it

```sh
bsl_reset -d io
```

if only want to  reset one myx

use bsl_reset
0x23-0x20 == 0x3, it is board id, 011
device id is from 0-7, use 1 as example
b 011 00001 == 0x61 == 97

```sh
bsl_reset -d io -i 97
```

## Example: how to reset MCU
---
use i2c-tools
0x1f is the i2c address

reset all

```sh
sudo i2cset -y 5 0x1f 0x01 0xff b
```

use bsl_reset

```sh
bsl_reset -d mcu
```

if only reset one myx

0x2 is the device id

```sh
sudo i2cset -y 5 0x1f 0x01 0x2 b
```

use bsl_reset

```sh
bsl_reset -d mcu -i 2
```
 