#!/bin/bash

# Setup environment
source ./setup_env.sh

sh scripts/stage_0.sh
# sh scripts/stage_0.sh

sh scripts/train.sh

sh scripts/stage_1.sh