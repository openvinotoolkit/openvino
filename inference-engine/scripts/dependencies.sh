#!/bin/bash -x

if [ "$1" = "" ]; then
		dep_arch=64
	else
		dep_arch=$1
fi

item_path=""
add_path() {
	component=$1
	item_path=""
    echo "Read file: dependencies_${dep_arch}.txt"
    grep_component="\b${component}\b"

	if [[ `cat dependencies_${dep_arch}.txt | grep $grep_component` ]];then
		archive_path=$(cat dependencies_${dep_arch}.txt | grep $grep_component | sed -E "s/${component}=//g")
		library_rpath=$(cat ld_library_rpath_${dep_arch}.txt | grep $grep_component | sed -E "s/${component}=//g")
		filename=$(basename $archive_path)
		if [[ (! -d $DL_SDK_TEMP/test_dependencies/$component/$filename) || (-d $DL_SDK_TEMP/test_dependencies/$component/$filename && ! $(ls -A $DL_SDK_TEMP/test_dependencies/$component/$filename)) ]]; then
			mkdir -p $DL_SDK_TEMP/test_dependencies/$component/$filename
			wget -q $archive_path -O $DL_SDK_TEMP/test_dependencies/$filename
			if [[ $filename == *.zip ]]; then
				unzip $DL_SDK_TEMP/test_dependencies/$filename -d $DL_SDK_TEMP/test_dependencies/$component/$filename
			elif [[ $filename == *.7z ]]; then
			    7za x -y $DL_SDK_TEMP/test_dependencies/$filename -o$DL_SDK_TEMP/test_dependencies/$component/$filename
			else
				tar xf $DL_SDK_TEMP/test_dependencies/$filename -C $DL_SDK_TEMP/test_dependencies/$component/$filename
			fi
			rm $DL_SDK_TEMP/test_dependencies/$filename
		fi
		item_path=$component/$filename/$library_rpath
	fi
}

runtimes=(MKL CLDNN MYRIAD GNA RH_Decoder DLIA OPENCV VPU_FIRMWARE_MA2450 VPU_FIRMWARE_MA2X8X HDDL OMP TBB AOCL_RTE)

export_library_path() {
    export LD_LIBRARY_PATH=$DL_SDK_TEMP/test_dependencies/$1:$LD_LIBRARY_PATH
}

export_env_variable() {
    export $2=$DL_SDK_TEMP/test_dependencies/$1
}

ma2480_path=""
for i in "${runtimes[@]}"
do
   add_path $i
   export_library_path $item_path
   if [ "$i" == "VPU_FIRMWARE_MA2X8X" ]
   then 
       ma2480_path=$item_path
   fi
   if [ "$i" == "HDDL" ]
   then
       cp -r $DL_SDK_TEMP/test_dependencies/$ma2480_path/* $DL_SDK_TEMP/test_dependencies/$item_path
       export HDDL_INSTALL_DIR=$DL_SDK_TEMP/test_dependencies/$item_path/..
   fi
done

echo DATA_PATH=$DATA_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/local/lib
