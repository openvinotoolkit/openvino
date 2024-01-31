function Choco-Install {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $PackageName,
        [string[]] $ArgumentList,
        [int] $RetryCount = 5
    )

    process {
        $count = 1
        while($true)
        {
            Write-Host "Running [#$count]: choco install $PackageName -y $ArgumentList"
            choco install $PackageName -y @ArgumentList

            $pkg = choco list --localonly $PackageName --exact --all --limitoutput
            if ($pkg) {
                Write-Host "Package installed: $pkg"
                break
            }
            else {
                $count++
                if ($count -ge $RetryCount) {
                    Write-Host "Could not install $PackageName after $count attempts"
                    exit 1
                }
                Start-Sleep -Seconds 30
            }
        }
    }
}
