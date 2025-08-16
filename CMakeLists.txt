cmake_minimum_required(VERSION 3.10)

project(GustGridVulkan)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(IS_WINDOWS FALSE)
set(IS_MACOS FALSE)
set(IS_LINUX FALSE)
if(WIN32)
    set(IS_WINDOWS TRUE)
elseif(APPLE)
    set(IS_MACOS TRUE)
else()
    set(IS_LINUX TRUE)
endif()

set(VK_SDK_PATH $ENV{VK_SDK_PATH})
if(IS_WINDOWS AND NOT VK_SDK_PATH)
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
elseif(IS_WINDOWS)
    include_directories(${VK_SDK_PATH}/Include)
    link_directories(${VK_SDK_PATH}/Lib)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/glm)
if(IS_WINDOWS)
    link_directories(${CMAKE_SOURCE_DIR}/include)
endif()

set(SOURCES
    src/main.cpp
    src/simulator.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

if(IS_MACOS)
    find_program(GLSLC_EXECUTABLE glslc
        HINTS
            ${VK_SDK_PATH}/macOS/bin
            /usr/local/bin
            $ENV{VULKAN_SDK}/bin
        PATHS
            /Users/$ENV{USER}/VulkanSDK/*/macOS/bin
    )
elseif(IS_WINDOWS)
    find_program(GLSLC_EXECUTABLE glslc
        HINTS
            ${VK_SDK_PATH}/Bin
            $ENV{VULKAN_SDK}/Bin
    )
else() # Linux
    find_program(GLSLC_EXECUTABLE glslc
        HINTS
            ${VK_SDK_PATH}/bin
            $ENV{VULKAN_SDK}/bin
            /usr/bin
            /usr/local/bin
    )
endif()

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found! Please install Vulkan SDK.")
endif()

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src/shaders/compiled)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/src/shaders/glsl/*.frag"
    "${CMAKE_SOURCE_DIR}/src/shaders/glsl/*.vert"
    "${CMAKE_SOURCE_DIR}/src/shaders/glsl/*.comp"
)

set(SPIRV_BINARY_FILES)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_SOURCE_DIR}/src/shaders/compiled/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSLC_EXECUTABLE} ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL}
        COMMENT "Compiling ${FILE_NAME}"
    )
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders
    DEPENDS ${SPIRV_BINARY_FILES}
    COMMENT "Compiling shaders"
)

add_dependencies(${PROJECT_NAME} Shaders)

if(IS_MACOS)
    find_package(Vulkan REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(Freetype REQUIRED)
    
    target_link_libraries(${PROJECT_NAME} 
        ${Vulkan_LIBRARIES} 
        glfw
        ${FREETYPE_LIBRARIES}
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${Vulkan_INCLUDE_DIRS} 
        ${FREETYPE_INCLUDE_DIRS}
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        VULKAN_HPP_NO_EXCEPTIONS
    )
elseif(IS_WINDOWS)
    find_library(Vulkan_LIBRARY vulkan-1 ${VK_SDK_PATH}/Lib)
    find_package(Freetype REQUIRED)

    target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} glfw3_mt ${FREETYPE_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${FREETYPE_INCLUDE_DIRS}
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        WIN32
    )
else() # Linux
    find_package(PkgConfig QUIET)
    find_package(Vulkan REQUIRED)
    find_package(Freetype REQUIRED)
    find_package(glfw3 3.3 QUIET)
    if(NOT glfw3_FOUND)
        if(PkgConfig_FOUND)
            pkg_check_modules(GLFW3 REQUIRED glfw3)
            add_library(glfw INTERFACE)
            target_include_directories(glfw INTERFACE ${GLFW3_INCLUDE_DIRS})
            target_link_libraries(glfw INTERFACE ${GLFW3_LIBRARIES})
        else()
            message(FATAL_ERROR "glfw3 not found. Please install it, e.g., sudo apt install libglfw3-dev")
        endif()
    endif()

    target_link_libraries(${PROJECT_NAME}
        Vulkan::Vulkan
    glfw
        ${FREETYPE_LIBRARIES}
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
    ${FREETYPE_INCLUDE_DIRS}
    $<IF:$<BOOL:${glfw3_FOUND}>,${glfw3_INCLUDE_DIRS},${GLFW3_INCLUDE_DIRS}>
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        VULKAN_HPP_NO_EXCEPTIONS
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(IS_MACOS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenMP QUIET openmp)
    endif()
    
    set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
    if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        target_compile_options(${PROJECT_NAME} PRIVATE 
            -Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include)
        target_link_libraries(${PROJECT_NAME} 
            ${LIBOMP_PREFIX}/lib/libomp.dylib)
        message(STATUS "Using Homebrew OpenMP at ${LIBOMP_PREFIX}")
    else()
        message(FATAL_ERROR "OpenMP not found - install with: brew install libomp")
    endif()
else()
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP found and linked")
    else()
        message(FATAL_ERROR "OpenMP not found")
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    message(STATUS "Building in Release mode")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
        )
        set_target_properties(${PROJECT_NAME} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -DNDEBUG
            -funroll-loops
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