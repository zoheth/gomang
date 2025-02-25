find . -type f -name "*-*" | while read file; do
    newname=$(echo "$file" | sed 's/-/_/g')
    if [ "$file" != "$newname" ]; then
        mv "$file" "$newname"
        echo "已重命名: $file -> $newname"
    fi
done
