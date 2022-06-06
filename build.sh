#!/bin/bash
if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    echo "Exiting .."
    return 1
fi
if [ -z "$1" ]; then
    echo "Argument is empty"
    echo "Exiting .."
    return 2
fi
if [ "$1" != "xgboost" ] && [ "$1" != "svr" ]; then
    echo "$1 is not a valid model type"
    echo "Model type must be one of: xgboost, svr"
    echo "Exiting .."
    return 3
fi

KONAN_BUILD_PATH="builds/$1"

# Copy over build files
mkdir -p "$KONAN_BUILD_PATH"
cp {Dockerfile,.dockerignore,Makefile} "$KONAN_BUILD_PATH/"
cp .konan.example "$KONAN_BUILD_PATH/.konan"

# Add tag to KONAN_APP_VERSION
sed -i "s/KONAN_APP_VERSION.*/&-$1/g" "$KONAN_BUILD_PATH/.konan"

# Copy over app files
mkdir -p "$KONAN_BUILD_PATH/app"
cp {retrain.sh,requirements.txt} "$KONAN_BUILD_PATH/app/"
# Copy over app/src files
mkdir -p "$KONAN_BUILD_PATH/app/src"
cp -r src/app/* "$KONAN_BUILD_PATH/app/src/"
# Copy app/src/utils files
mkdir -p "$KONAN_BUILD_PATH/app/src/utils"
cp -r src/playground/utils/* "$KONAN_BUILD_PATH/app/src/utils"

# Copy over app/artifacts files
mkdir -p "$KONAN_BUILD_PATH/app/artifacts"
KONAN_MODEL_REGRESSOR_NAME="$1" KONAN_MODEL_ARTIFACTS_PATH="$KONAN_BUILD_PATH/app/artifacts" python src/playground/train.py

pushd "$KONAN_BUILD_PATH"
make release
popd
