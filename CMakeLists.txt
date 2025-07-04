cmake_minimum_required(VERSION 3.10)

project(GustGridVulkan)

set(IS_MACOS FALSE)
if(APPLE)
    set(IS_MACOS TRUE)
endif()

set(VK_SDK_PATH $ENV{VK_SDK_PATH})
if(NOT VK_SDK_PATH AND NOT IS_MACOS)
    message(FATAL_ERROR "VK_SDK_PATH environment variable is not set. Please set it to the path of your Vulkan SDK.")
endif()

if(IS_MACOS)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
    if(NOT VK_SDK_PATH)
        find_package(Vulkan REQUIRED)
        include_directories(${Vulkan_INCLUDE_DIRS})
    else()
        include_directories(${VK_SDK_PATH}/Include)
        link_directories(${VK_SDK_PATH}/macOS/lib)
    endif()
else()
    include_directories(${VK_SDK_PATH}/Include)
    link_directories(${VK_SDK_PATH}/Lib)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/shaders/compiled/main.vert.spv
    src/shaders/compiled/main.frag.spv
)

add_executable(${PROJECT_NAME} ${SOURCES})

if(IS_MACOS)
    find_package(Vulkan REQUIRED)
    find_package(glfw3 REQUIRED)
    
    target_link_libraries(${PROJECT_NAME} 
        ${Vulkan_LIBRARIES} 
        glfw
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        VULKAN_HPP_NO_EXCEPTIONS
    )
else()
    find_library(Vulkan_LIBRARY vulkan-1 ${VK_SDK_PATH}/Lib)
    target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} glfw3_mt)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        WIN32
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    message(STATUS "Building in Release mode")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME} PRIVATE 
            -O3
            -DNDEBUG
            -funroll-loops
            -fvectorize
            -fslp-vectorize
            -ffast-math
            -march=native
            -mtune=native
            -flto
            -fomit-frame-pointer
            -finline-functions
            -ftree-vectorize
        )
        set_target_properties(${PROJECT_NAME} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT_NAME} PRIVATE 
            /O2
            /Oi
            /Ot
            /Gy
            /GL
            /fp:fast
            /DNDEBUG
        )
        set_target_properties(${PROJECT_NAME} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()
else()
    message(STATUS "Building in Debug mode")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME} PRIVATE 
            -Wall -Wextra -g -O0
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT_NAME} PRIVATE 
            /W4 /Zi /Od
        )
    endif()
endif()

if(IS_MACOS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@executable_path/../lib"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin/lib
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libraries to bin directory..."
    )
endif()