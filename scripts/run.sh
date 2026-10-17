#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

set -e

# rm -rf example_data/trendify

# trendify run \
#     -i "example_data/models/*" \
#     -g matrix_generator.py:build_configuration_matrix \
#     -o example_data/trendify \
#     -n 4

trendify viewer \
    example_data/trendify/trendify.db \
    --host 0.0.0.0 \
    --port 8084
