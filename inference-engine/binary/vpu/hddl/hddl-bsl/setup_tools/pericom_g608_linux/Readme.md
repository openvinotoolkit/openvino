# Set Pericom Switch for HDDL-R

This is a script dedicated for some Pericom Switch on HDDL-R

## Prerequisites

Please make you have functional `lspci` and `setpci`
If not, you can get them by

```sh
sudo apt install pciutils # On Ubuntu
sudo yum install pciutils # On CentOS
```

## Targeting devices

If you have some `12d8:2608` Pericom Switches in your system,  
and no myx device showes up after boot  
You can try to fix it with this script

## Installation

Known to be functional on Ubuntu 16.04.4 or later  
You can just install the startup script by

```sh
sudo ./install.sh
```

CentOS is supposed to be supported  
`install.sh` is not tested yet

## Notice

This script __unsets__ `0x224 bit 18` for all downstream ports of every Pericom switch  
And it will __NOT__ check whether it is for hddl
