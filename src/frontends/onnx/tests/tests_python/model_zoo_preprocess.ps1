# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# default ONNX Model Zoo commit hash ID:
$ONNX_SHA="5faef4c33eba0395177850e1e31c4a6a9e634c82".Substring(0, 8)
$MODELS_DIR="."
$ENABLE_ONNX_MODELS_ZOO=$false
$ENABLE_MSFT_MODELS=$false
$FORCE_MODE=$false

function print_help() {
    write("Model preprocessing options:")
    write("    -h display this help message")
    write("    -d <DIR> set location of the models (for onnx model ZOO and MSFT models)")
    write("    By default the models location is current folder (./model_zoo)")
    write("    -o update Onnx Model Zoo models")
    write("    -s Onnx Model Zoo commit SHA")
    write("    -m update MSFT models")
    write("    -f force update of a chosen model")
    write("")
    write("Note: This script requires wget, GNU tar (not bsdtar) and git with LFS support.")
}

for ($i = 0; $i -lt $args.count; $i++) {
    switch($args[$i]) {
        "-h" {
            print_help
        }
        "-?" {
            print_help
        }
        # Windows-friendly help
        "/?" {
            print_help
        }
        "-d" {
            if($i + 1 -ge $args.count) {
                print_help
            } else {
                $MODELS_DIR = $args[$i+1]
                $i++
            }
        }
        "-o" {
            $ENABLE_ONNX_MODELS_ZOO=$true
        }
        "-s" {
            if($i + 1 -ge $args.count) {
                print_help
            } else {
                $ONNX_SHA = $args[$i+1]
                $i++
            }
        }
        "-m" {
            $ENABLE_MSFT_MODELS=$true
        }
        "-f" {
            $FORCE_MODE=$true
        }
    }
}

$MODEL_ZOO_DIR="$MODELS_DIR/model_zoo"
$ONNX_MODELS_DIR="$MODEL_ZOO_DIR/onnx_model_zoo_$ONNX_SHA"
$MSFT_MODELS_DIR="$MODEL_ZOO_DIR/MSFT"

function pull_and_postprocess_onnx_model_zoo() {
    Push-Location
    Set-Location -Path "$ONNX_MODELS_DIR"

    & git fetch
    & git reset HEAD --hard

    & git checkout -f "$ONNX_SHA"

    write("Pulling models data via Git LFS for onnx model zoo repository")
    & git lfs pull --include="*" --exclude="*.onnx"

    Get-ChildItem * -Include *.onnx -Recurse | Remove-Item

    Foreach ($file in Get-ChildItem * -Include *.tar.gz -Recurse) {
        $OutPath = $file.DirectoryName + "/" + [System.IO.Path]::GetFileNameWithoutExtension($file.BaseName)
        if(Test-Path -Path $OutPath -PathType Container) {
            Remove-Item $OutPath -Recurse
        }
        New-Item -Path $OutPath -ItemType Directory
        $Arch = $file.FullName
        & "C:/Program Files/7-zip/7z.exe" x "$($Arch)" -o".\tmpdir"
        Foreach($tar in Get-ChildItem ".\tmpdir\*" -Include *.tar) {
            & "C:/Program Files/7-zip/7z.exe" x "$($tar.FullName)" -o"$($OutPath)"
        }
        Remove-Item -Path ".\tmpdir\*" -Recurse
    }

    write("Postprocessing of ONNX Model Zoo models:")

    write("Fix roberta model")
    Push-Location
    Set-Location "./text/machine_comprehension/roberta/model/roberta-sequence-classification-9/roberta-sequence-classification-9"
    New-Item -Path "test_data_set_0" -ItemType Container
    Foreach($file in Get-ChildItem * -Include *.pb) {
        Move-Item -Path $file.FullName -Destination "test_data_set_0\"
    }
    Pop-Location

    Pop-Location
}

function update_onnx_models() {
    if(-not(Test-Path -Path "$ONNX_MODELS_DIR" -PathType Container)) {
        git clone --progress "https://github.com/onnx/models.git" "$ONNX_MODELS_DIR"
    } else {
        $git_remote_url = git -C "$ONNX_MODELS_DIR" config --local remote.origin.url
        write($git_remote_url)
        if($git_remote_url -eq "https://github.com/onnx/models.git") {
            write("The proper github repository detected: $git_remote_url")
        } else {
            write("The ONNX Model Zoo repository doesn't exist then will be cloned")
            git clone --progress "https://github.com/onnx/models.git" "$ONNX_MODELS_DIR"
        }
    }

    if(Test-Path -Path "$ONNX_MODELS_DIR\tmpdir" -PathType Container) {
        Remove-Item -Path "$ONNX_MODELS_DIR\tmpdir" -Recurse
    }
    New-Item -Path "$ONNX_MODELS_DIR\tmpdir" -ItemType Directory

    pull_and_postprocess_onnx_model_zoo
}

function update_msft_models() {
    Invoke-WebRequest -Uri "https://onnxruntimetestdata.blob.core.windows.net/models/20191107.zip" -OutFile "$MSFT_MODELS_DIR.zip"
    Expand-Archive -Path "$MSFT_MODELS_DIR.zip" -DestinationPath "$MSFT_MODELS_DIR"
    if(Test-Path -Path "$MSFT_MODELS_DIR" -PathType Container) {
        Remove-Item "$MSFT_MODELS_DIR.zip"
    }
}

function postprocess_msft_models() {
    write("Postprocessing of MSFT models:")

    write("Fix LSTM_Seq_lens_unpacked")
    Rename-Item -Path "$MSFT_MODELS_DIR/opset9/LSTM_Seq_lens_unpacked/seq_lens_sorted" -NewName "$MSFT_MODELS_DIR/opset9/LSTM_Seq_lens_unpacked/test_data_set_0"
    Rename-Item -Path "$MSFT_MODELS_DIR/opset9/LSTM_Seq_lens_unpacked/seq_lens_unsorted" -NewName "$MSFT_MODELS_DIR/opset9/LSTM_Seq_lens_unpacked/test_data_set_1"
}

if(-not(Test-Path -Path "$MODEL_ZOO_DIR" -PathType Container)) {
    write("The general model directory: $MODEL_ZOO_DIR doesn't exist on your filesystem, it will be created")
    New-Item -Path "$MODEL_ZOO_DIR" -ItemType Directory
} else {
    write("The general model directory: $MODEL_ZOO_DIR found")
}

if($ENABLE_ONNX_MODELS_ZOO -eq $false -and $ENABLE_MSFT_MODELS -eq $false) {
    write("Please choose an option to update chosen model:")
    write("    -o to update ONNX Model ZOO")
    write("    -m to update MSFT models")
}

if($ENABLE_ONNX_MODELS_ZOO -eq $true) {
    if($FORCE_MODE -eq $true) {
        Remove-Item -Path "$ONNX_MODELS_DIR" -Recurse
    }
    update_onnx_models
}

if($ENABLE_MSFT_MODELS -eq $true) {
    if($FORCE_MODE -eq $true) {
        Remove-Item -Path "$MSFT_MODELS_DIR" -Recurse
    }
    update_msft_models
}
