#!/bin/bash

# If true, print info
VERBOSE=true

CONDAENV="/opt/conda/envs/growsp_forms" # <- for docker install
#CONDAENV="~/miniconda3/envs/growsp_forms" <- for local install (may vary)
echo $CONDAENV
# Find python version
VERSION=$(python --version 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$VERSION" ]]
then
    echo "No python installation detected on your system!" 
fi
# Parse python version
readarray -td '' version_array < <(awk '{ gsub(/\./,"\0"); print; }' <<<"$VERSION."); unset 'version_array[-1]';
PYTHON_VERSION="${version_array[0]}.${version_array[1]}"
if [ "$VERBOSE" = true ]
then
    echo "Using python version $PYTHON_VERSION"
fi
# This whole step is necessary only because sometimes the directory name is python3.<x>m instead of just python3.<x>
OUTPUT=$(python$PYTHON_VERSION-config --includes --libs)
readarray -td '' path_list < <(awk '{ gsub(/[ \t]+/,"\0"); gsub(/[-\n]/,""); print; }' <<<"$OUTPUT "); unset 'path_list[-1]';

PYNAME="None"
for path in ${path_list[*]}
do
    result=$(basename "$path")
    if [[ $result == "python$PYTHON_VERSION"* ]]
    then
        PYNAME="$result"
    fi
done

# Add path
PATH_TO_ADD="$CONDAENV/include/$PYNAME"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PATH_TO_ADD"
if [ "$VERBOSE" = true ]
then
    echo "Added path $PATH_TO_ADD"
fi

# Cmake
rm -r build
mkdir build
cd build
if [ "$VERBOSE" = true ]
then
    echo "Creating makefiles..."
fi
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/lib$PYNAME.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/$PYNAME -DBOOST_INCLUDEDIR=$CONDAENV/include  -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
if [ "$VERBOSE" = true ]
then
    echo "Building..."
fi
make