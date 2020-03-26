#! /bin/bash

DRIVER_DIR=`dirname "$(readlink -f "$0")"`
COMPONENTS="
drv_ion
drv_vsc
"

if [ "$1" = "install" ]; then
  TARGET=$1
elif [ "$1" = "uninstall" ]; then
  TARGET=$1
else
  echo "Unrecognezed argements. Please use"
  echo "    bash setup.sh install|uninstall"
  exit
fi

function install_component {
  local component=$1
  local target=$2

  echo "Running $target for component $component"
  cd $DRIVER_DIR/$component
  make $target
}

for component in $COMPONENTS
do
  install_component $component $TARGET
done
