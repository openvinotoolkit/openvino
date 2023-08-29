#!/bin/sh

source_dir=$1
output_dir=$2

for file in $(find "$source_dir" -iname "*.md" -type f ! -path "*/thirdparty/*" ! -path "*/bin/*" ! -path "*/tests/*" ! -path "*/temp/*" ! -path "*/docs/ops/internal/*")
do
    firstline=$(head -n1 "$file")
    if (echo "$firstline" | grep -qo '{#.*}'); then
        echo "Label found in $file file. Processing with rst conversion..."
        out_filename=$(echo "$firstline" | sed -e 's/.*{#\(.*\)}.*/\1/')
        out_path="${output_dir}/${out_filename}.rst"
        pandoc --from=markdown --to=rst --output="$out_path" $file
    else
        echo "Skipped $file"
    fi
done