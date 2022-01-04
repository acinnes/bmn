cmake_minimum_required(VERSION 3.17)
project(playbmn LANGUAGES CXX CUDA)

find_package(CUDA)
if(CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(KALDI_CUDA_NVCC_FLAGS "--default-stream=per-thread;-std=c++${CMAKE_CXX_STANDARD}")
    if(MSVC)
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler /permissive-,/FS,/wd4819,/EHsc,/bigobj")
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler /wd4018,/wd4244,/wd4267,/wd4291,/wd4305")
    else()
    #     list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler -std=c++${CMAKE_CXX_STANDARD}")
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
    endif()
    set(CUDA_NVCC_FLAGS ${KALDI_CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS})

    add_definitions(-DUSE_CUDA=1)
    add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_curand_LIBRARY}
    )
endif()

# add the executable
add_executable(playbmn playbmn.cpp)
