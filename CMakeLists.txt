cmake_minimum_required(VERSION 3.10)

project(GustGridVulkan)

set(VK_SDK_PATH $ENV{VK_SDK_PATH})
if(NOT VK_SDK_PATH)
    message(FATAL_ERROR "VK_SDK_PATH environment variable is not set. Please set it to the path of your Vulkan SDK.")
endif()

include_directories(${VK_SDK_PATH}/Include)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/shaders/main.vert
    src/shaders/main.frag
    src/shaders/ui.vert
    src/shaders/ui.frag
    src/shaders/volume.vert
    src/shaders/volume.frag
    src/shaders/tempreader.glsl
    src/shaders/text.frag
)

add_executable(${PROJECT_NAME} ${SOURCES})

find_library(Vulkan_LIBRARY vulkan-1 ${VK_SDK_PATH}/Lib)
target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} glfw3_mt)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DWIN32)
endif()