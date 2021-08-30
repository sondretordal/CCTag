# Install Pybind11
vcpkg install pybind11 --triplet x64-windows

# Activate python env before configuring CMake project
conda activate base

# Configure CMake Project
cmake .. -G "Visual Studio 16 2019" -A x64 -D CMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -D CCTAG_WITH_CUDA=OFF -D CMAKE_INSTALL_PREFIX=C:/Users/sondr/GitHub/CCTag/install

# Build and Install
cmake --build . --config Release --target INSTALL