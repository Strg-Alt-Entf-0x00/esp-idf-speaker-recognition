cmake_minimum_required(VERSION 3.16)

# Register extra component directories
set(EXTRA_COMPONENT_DIRS
    "../../"
    "components/board_audio"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ws-esp32-p4-wifi6-console-sr-dev)
