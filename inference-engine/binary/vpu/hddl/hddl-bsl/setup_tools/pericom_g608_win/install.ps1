if ([environment]::Is64BitOperatingSystem) {
    $url = "https://www.diodes.com/assets/Uploads/disableacs64.zip"
    $zip_name = "disableacs64.zip"
    $exe_name = "disableacs64.exe"
}
else {
    $url = "https://www.diodes.com/assets/Uploads/disableacs.zip"
    $zip_name = "disableacs.zip"
    $exe_name = "disableacs.exe"
}

$zip_path = "$PSScriptRoot\$zip_name"
$exe_path = "$PSScriptRoot\$exe_name"
$target_path = "C:\pericom_g608.exe"

function suggest_download {
    Write-Output "Download fail"
    Write-Output "Please download $url to the folder containing this script"
    pause
    exit
}

function download_zip {
    if (Test-Path $zip_path -PathType Leaf) {
        Write-Output "$zip_name found, skip download"
    }
    else {
        Write-Output "Downloading $url"
        Invoke-WebRequest -Uri $url -OutFile $zip_path

        if (!(Test-Path $zip_path -PathType Leaf)) {
            suggest_download
        }
    }
}

function unzip {
    Expand-Archive -Path $zip_path -DestinationPath $PSScriptRoot
}

function get_exe_file {
    if (Test-Path $exe_path -PathType Leaf) {
        Write-Output "$exe_name found, skip download"
    }
    else {
        download_zip
        unzip
    }
}

function set_task {
    $command = "
        echo 'Copying from $exe_path to $target_path'
        copy $exe_path $target_path

        echo 'Setting up Schedualed Task'
        schtasks /create /sc ONSTART /tn 'HDDL Set Pericom G608' /tr $target_path /it /ru System /f
        $target_path
        pause
    " 
    Start-Process PowerShell -ArgumentList $command -Wait -Verb RunAs
}

function main {
    get_exe_file
    set_task
}

main
