#!/usr/bin/env bash
# File    : t5tokenize.sh
# Brief   : Creates Python virtual environment and runs t5tokenize.py inside it.
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : May 2, 2024
# Repo    : https://github.com/martin-rizzo/PoweredT5Encoder
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                             Powered T5 Encoder
#       An enhanced T5 encoder with integrated weighted prompt support
#
#     Copyright (c) 2024 Martin Rizzo
#
#     Permission is hereby granted, free of charge, to any person obtaining
#     a copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so, subject to
#     the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
ScriptName=${BASH_SOURCE[0]##*/}                     # name of the script file
ScriptDir=$(realpath $(dirname "${BASH_SOURCE[0]}")) # path to the dir containing the script
DistroName=               # name of the linux distribution (empty=auto-detect)
PackageManager=           # package manager to use (empty=auto-detect)
CompatiblePython=         # command for a compatible python version (empty=auto-detect)
VEnvDir="$ScriptDir/venv" # directory for the python virtual environment
ShowVersion=false
Reinstall=false

# import functions for auto-detection and virtual environment management
source "$ScriptDir/xtras/utils.sh"

# check if the user requests version or reinstall, and set the respective flags
[[ $# -eq 1 && $1 == '--version'   ]] && ShowVersion=true
[[ $# -eq 1 && $1 == '--reinstall' ]] && Reinstall=true

# configure the 'DistroName', 'PackageManager', 'CompatiblePython' variables
autodetect_distro_vars

# ensure the python version in 'CompatiblePython' is installed on the system
require_system_command "$CompatiblePython"

# if the user requests a reinstall, remove the virtual environment directory
if [[ $Reinstall == true ]]; then
    if [[ $ScriptDir == /* && "$VEnvDir" =~ ^"$ScriptDir"/.* ]]; then
        rm -rf "$VEnvDir"
    fi
fi

# create the virtual environment with all required libraries
if [[ ! -e $VEnvDir ]]; then
    virtual_python "$VEnvDir" !pip install --upgrade pip
    virtual_python "$VEnvDir" !pip install -r "$ScriptDir/requirements.txt"
fi

# if the user requested, show info and version of everything
if [[ $ShowVersion == true ]]; then
    virtual_python "$VEnvDir" !pip install --upgrade pip >/dev/null
    virtual_python "$VEnvDir" !pip install -r "$ScriptDir/requirements.txt" >/dev/null

    # get the version of the 't5tokenize.py' script
    script_version=$(virtual_python "$VEnvDir" "$ScriptDir/t5tokenize.py" --version)
    pip_version=$(virtual_python "$VEnvDir" !pip --version)

    echo "venv-dir : $VEnvDir"
    echo "python   : ./$CompatiblePython"
    echo
    echo "$($CompatiblePython --version)"
    echo "${pip_version:0:10}"
    # print the version of 'transformers' installed in the virtual environment
    virtual_python "$VEnvDir" !python -c "import transformers;  print('transformers' , transformers.__version__ )"
    virtual_python "$VEnvDir" !python -c "import sentencepiece; print('sentencepiece', sentencepiece.__version__)"
    echo "$script_version"
    echo
    exit 0
fi

# run the Python script using the virtual environment
virtual_python "$VEnvDir" "$ScriptDir/t5tokenize.py" "$@"
