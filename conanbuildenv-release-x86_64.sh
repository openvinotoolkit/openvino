script_folder="/home/vyomesh/gsoc_Proj/openvino"
echo "echo Restoring environment" > "$script_folder/deactivate_conanbuildenv-release-x86_64.sh"
for v in PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PKG_CONFIG ACLOCAL_PATH AUTOMAKE_CONAN_INCLUDES
do
   is_defined="true"
   value=$(printenv $v) || is_defined="" || true
   if [ -n "$value" ] || [ -n "$is_defined" ]
   then
       echo export "$v='$value'" >> "$script_folder/deactivate_conanbuildenv-release-x86_64.sh"
   else
       echo unset $v >> "$script_folder/deactivate_conanbuildenv-release-x86_64.sh"
   fi
done

export PATH="/home/vyomesh/.conan2/p/b/flatb8c4702139cc3e/p/bin:/home/vyomesh/.conan2/p/b/protoa8c96a7fd1a77/p/bin:/home/vyomesh/.conan2/p/patch7e697dcc93fe0/p/bin:/home/vyomesh/.conan2/p/pkgcoa08771b75b56d/p/bin:/home/vyomesh/.conan2/p/cmake9d32db761075c/p/bin:$PATH"
export LD_LIBRARY_PATH="/home/vyomesh/.conan2/p/b/flatb8c4702139cc3e/p/lib:/home/vyomesh/.conan2/p/b/protoa8c96a7fd1a77/p/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/home/vyomesh/.conan2/p/b/flatb8c4702139cc3e/p/lib:/home/vyomesh/.conan2/p/b/protoa8c96a7fd1a77/p/lib:$DYLD_LIBRARY_PATH"
export PKG_CONFIG="/home/vyomesh/.conan2/p/pkgcoa08771b75b56d/p/bin/pkgconf"
export ACLOCAL_PATH="/home/vyomesh/.conan2/p/pkgcoa08771b75b56d/p/bin/aclocal:$ACLOCAL_PATH"
export AUTOMAKE_CONAN_INCLUDES="/home/vyomesh/.conan2/p/pkgcoa08771b75b56d/p/bin/aclocal:$AUTOMAKE_CONAN_INCLUDES"