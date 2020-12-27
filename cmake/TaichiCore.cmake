set(CORE_LIBRARY_NAME taichi_core)

option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_CUDA "Build with the CUDA backend" ON)
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)
option(TI_WITH_CC "Build with the C backend" OFF)
option(TI_WITH_VULKAN "Build with the Vulkan backend" ON)

if(UNIX AND NOT APPLE)
    # Handy helper for Linux
    # https://stackoverflow.com/a/32259072/12003165
    set(LINUX TRUE)
endif()

if (APPLE)
    if (TI_WITH_CUDA)
        set(TI_WITH_CUDA OFF)
        message(WARNING "CUDA backend not supported on OS X. Setting TI_WITH_CUDA to OFF.")
    endif()
    if (TI_WITH_OPENGL)
        set(TI_WITH_OPENGL OFF)
        message(WARNING "OpenGL backend not supported on OS X. Setting TI_WITH_OPENGL to OFF.")
    endif()
    if (TI_WITH_CC)
        set(TI_WITH_CC OFF)
        message(WARNING "C backend not supported on OS X. Setting TI_WITH_CC to OFF.")
    endif()
    if (TI_WITH_VULKAN)
        set(TI_WITH_VULKAN OFF)
        message(WARNING "Vulkan backend not supported on OS X. Setting TI_WITH_VULKAN to OFF.")
    endif()
endif()

if (WIN32)
    if (TI_WITH_CC)
        set(TI_WITH_CC OFF)
        message(WARNING "C backend not supported on Windows. Setting TI_WITH_CC to OFF.")
    endif()
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c")
    set(TI_WITH_OPENGL OFF)
    message(WARNING "external/glad submodule not detected. Settings TI_WITH_OPENGL to OFF.")
endif()

file(GLOB TAICHI_CORE_SOURCE
        "taichi/*/*/*/*.cpp" "taichi/*/*/*.cpp" "taichi/*/*.cpp" "taichi/*.cpp"
        "taichi/*/*/*/*.h" "taichi/*/*/*.h" "taichi/*/*.h" "taichi/*.h" "tests/cpp/*.cpp")

file(GLOB TAICHI_BACKEND_SOURCE "taichi/backends/**/*.cpp" "taichi/backends/**/*.h")

file(GLOB TAICHI_CPU_SOURCE "taichi/backends/cpu/*.cpp" "taichi/backends/cpu/*.h")
file(GLOB TAICHI_CUDA_SOURCE "taichi/backends/cuda/*.cpp" "taichi/backends/cuda/*.h")
file(GLOB TAICHI_METAL_SOURCE "taichi/backends/metal/*.h" "taichi/backends/metal/*.cpp" "taichi/backends/metal/shaders/*")
file(GLOB TAICHI_OPENGL_SOURCE "taichi/backends/opengl/*.h" "taichi/backends/opengl/*.cpp" "taichi/backends/opengl/shaders/*")
file(GLOB TAICHI_CC_SOURCE "taichi/backends/cc/*.h" "taichi/backends/cc/*.cpp")
file(GLOB TAICHI_VULKAN_SOURCE "taichi/backends/vulkan/*.h" "taichi/backends/vulkan/*.cpp")

list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_BACKEND_SOURCE})

list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CPU_SOURCE})

if (TI_WITH_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CUDA_SOURCE})
endif()

if(NOT CUDA_VERSION)
    set(CUDA_VERSION 10.0)
endif()

# TODO(#529) include Metal source only on Apple MacOS, and OpenGL only when TI_WITH_OPENGL is ON
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_METAL_SOURCE})
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_OPENGL_SOURCE})

if (TI_WITH_OPENGL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_OPENGL")
  # Q: Why not external/glad/src/*.c?
  # A: To ensure glad submodule exists when TI_WITH_OPENGL is ON.
  file(GLOB TAICHI_GLAD_SOURCE "external/glad/src/glad.c")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_GLAD_SOURCE})
endif()

if (TI_WITH_CC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CC")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CC_SOURCE})
endif()

# This compiles all the libraries with -fPIC, which is critical to link a static
# library into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The short-term goal is to have a sub-library that is mostly Taichi-focused,
# free from the "application" layer such as pybind11 or GUI. At a minimum, we
# must decouple from pybind11/python-environment. This sub-lib will then be
# unit testtable.
# TODO(#2198): Long-term speaking, we should create a separate library for each
# sub-module. This way we can guarantee that the lib dependencies form a DAG.
file(GLOB TAICHI_TESTABLE_SRC
      "taichi/common/*.cpp"
      "taichi/common/*.h"
      "taichi/ir/ir_builder.*"
      "taichi/ir/ir.*"
      "taichi/ir/offloaded_task_type.*"
      "taichi/ir/snode_types.*"
      "taichi/ir/snode.*"
      "taichi/ir/statements.*"
      "taichi/ir/type_factory.*"
      "taichi/ir/type_utils.*"
      "taichi/ir/type.*"
      "taichi/transforms/statement_usage_replace.cpp"
      "taichi/program/arch.*"
      "taichi/program/compile_config.*"
      "taichi/system/timer.*"
      "taichi/system/profiler.*"
)

if (TI_WITH_VULKAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_VULKAN")
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_VULKAN_SOURCE})
endif()

# TODO(#2196): Maybe we can do the following renaming in the end?
# taichi_core --> taichi_pylib (this requires python-side refactoring...)
# taichi_testable_lib --> taichi_core
set(TAICHI_TESTABLE_LIB taichi_testable_lib)
add_library(${TAICHI_TESTABLE_LIB} STATIC ${TAICHI_TESTABLE_SRC})

list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_TESTABLE_SRC})

add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE})
target_link_libraries(${CORE_LIBRARY_NAME} ${TAICHI_TESTABLE_LIB})

if (APPLE)
    # Ask OS X to minic Linux dynamic linking behavior
    target_link_libraries(${CORE_LIBRARY_NAME} "-undefined dynamic_lookup")
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(external/include)
include_directories(external/spdlog/include)
if (TI_WITH_OPENGL)
  include_directories(external/glad/include)
endif()

set(LIBRARY_NAME ${CORE_LIBRARY_NAME})

if (TI_WITH_OPENGL)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

  message("Building with GLFW")
  add_subdirectory(external/glfw)
  target_link_libraries(${LIBRARY_NAME} glfw)
endif()

# http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
if(${LLVM_PACKAGE_VERSION} VERSION_LESS "10.0")
    message(FATAL_ERROR "LLVM version < 10 is not supported")
endif()
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
message("LLVM include dirs ${LLVM_INCLUDE_DIRS}")
message("LLVM library dirs ${LLVM_LIBRARY_DIRS}")
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs
        Core
        ExecutionEngine
        InstCombine
        OrcJIT
        RuntimeDyld
        TransformUtils
        BitReader
        BitWriter
        Object
        ScalarOpts
        Support
        native
        Linker
        Target
        MC
        Passes
        ipo
        Analysis
        )
target_link_libraries(${LIBRARY_NAME} ${llvm_libs})

if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    llvm_map_components_to_libnames(llvm_aarch64_libs AArch64)
    target_link_libraries(${LIBRARY_NAME} ${llvm_aarch64_libs})
endif()

if (TI_WITH_CUDA)
    llvm_map_components_to_libnames(llvm_ptx_libs NVPTX)
    target_link_libraries(${LIBRARY_NAME} ${llvm_ptx_libs})
endif()

if (TI_WITH_VULKAN)
    # Vulkan libs
    # https://cmake.org/cmake/help/latest/module/FindVulkan.html
    # https://github.com/PacktPublishing/Learning-Vulkan/blob/master/Chapter%2003/HandShake/CMakeLists.txt
    find_package(Vulkan REQUIRED)
    message(STATUS "Vulkan_INCLUDE_DIR=${Vulkan_INCLUDE_DIR}")
    message(STATUS "Vulkan_LIBRARY=${Vulkan_LIBRARY}")
    include_directories(${Vulkan_INCLUDE_DIR})
    target_link_libraries(${CORE_LIBRARY_NAME} ${Vulkan_LIBRARY})

    # shaderc libs
    # TODO: Is there a better way to auto detect this?
    set(SHADERC_ROOT_DIR "/home/yekuang/Libs/shaderc-linux-gcc/install")
    if (NOT SHADERC_ROOT_DIR)
        message(FATAL_ERROR
            "Please specify `-DSHADERC_ROOT_DIR=/path/to/shaderc` for developing the Vulkan backend. "
            "The path should be the root direcotry containing `includes`, `lib` and `bin`.\n"
            "If you haven't installed `shaderc`, please visit\n"
            "https://github.com/google/shaderc/blob/main/downloads.md\n"
            "to download the matching libraries.")
    endif()
    find_library(SHADERC_LIB NAMES "shaderc_combined" PATHS "${SHADERC_ROOT_DIR}/lib" REQUIRED)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE "${SHADERC_ROOT_DIR}/include")
    target_link_libraries(${CORE_LIBRARY_NAME} ${SHADERC_LIB})
    if (LINUX)
        # shaderc requires pthread
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${CORE_LIBRARY_NAME} Threads::Threads)
    endif()
endif ()

# Optional dependencies

if (APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} "-framework Cocoa -framework Metal")
endif ()

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread stdc++)
    if (APPLE)
        # OS X
    else()
        # Linux
        target_link_libraries(${CORE_LIBRARY_NAME} stdc++fs X11)
        target_link_libraries(${CORE_LIBRARY_NAME} -static-libgcc -static-libstdc++)
        target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/misc/linker.map)
        target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--wrap=log2f) # Avoid glibc dependencies
    endif()
else()
    # windows
    target_link_libraries(${CORE_LIBRARY_NAME} Winmm)
endif ()
message("PYTHON_LIBRARIES: " ${PYTHON_LIBRARIES})

foreach (source IN LISTS TAICHI_CORE_SOURCE)
    file(RELATIVE_PATH source_rel ${CMAKE_CURRENT_LIST_DIR} ${source})
    get_filename_component(source_path "${source_rel}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()

if (MSVC)
    set_property(TARGET ${CORE_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

if (WIN32)
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()
