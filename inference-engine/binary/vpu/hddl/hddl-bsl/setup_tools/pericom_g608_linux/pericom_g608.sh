#! /bin/bash

DEBUG=0

for arg in "$@"; do
	if (("$arg" == "-d")); then
		echo "Debug Mode"
		DEBUG=1
	fi
done

VIDPID="12d8:2608"
UPSTREAM_BUSES=()
BIT_MASK=$((1 << 18 ^ 0xffffffff))

set_device() {
	device=$1
	echo "Setting $(lspci -s $device)"
	orig_value="0x$(setpci -s $device 224.L)"
	echo "Orig value of 0x224 is $orig_value"
	target_value=$(printf "%x" $((BIT_MASK & orig_value)))
	echo "Target value of 0x224 is $target_value"

	if (($DEBUG)); then
		echo "Setting $device 0x224 to $target_value"
	else
		setpci -s $device 224.L=$target_value
	fi
}

try_with_bus() {
	BUS=$1
	echo "Checking devices on bus $BUS:xx.x"
	DEVICE_LIST=$(lspci -d $VIDPID -s $BUS: | awk '{print $1}')
	COUNT=$(echo "$DEVICE_LIST" | wc -l)
	if (($COUNT == 1)); then
		echo "Upstream bus detected on $BUS"
		UPSTREAM_BUSES+=("$BUS")
	else
		echo "Downstream bus detected"
		for device in $DEVICE_LIST; do
			set_device $device
		done
	fi
}

get_possible_buses() {
	AWK_CMD='{
	slots=$1
	split(slots,bus,":")
	print bus[1]
}'
	echo "Scanning for $VIDPID devices"
	PCI_LIST=$(lspci -d $VIDPID)
	BUSES=$(echo "$PCI_LIST" | awk "$AWK_CMD" | sort -u)
}

trigger_rescan() {
	for upstream_bus in "${UPSTREAM_BUSES[@]}"; do
		echo "Removing Upstream port on $upstream_bus"
		if ((!$DEBUG)); then
			echo 1 >/sys/bus/pci/devices/0000\:$upstream_bus\:00.0/remove
		fi
	done
	echo "Triggering PCI rescan"
	if ((!$DEBUG)); then
		echo 1 >/sys/bus/pci/rescan
	fi
}

main() {
	get_possible_buses
	for bus in $BUSES; do
		try_with_bus $bus
	done
	trigger_rescan
}

main
