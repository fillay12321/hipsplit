cmake_minimum_required(VERSION 3.21)
project(hipsplit LANGUAGES CXX VERSION 0.1.0)

# The planning core is deliberately HIP-free so it builds and tests anywhere.
# The HIP-backed device query / occupancy / executor layer is built only when a
# ROCm/HIP install is actually present.
option(HIPSPLIT_BUILD_TESTS  "Build hipsplit unit tests" ON)
option(HIPSPLIT_WITH_HIP     "Build the HIP-backed device query/occupancy layer" ON)
option(HIPSPLIT_BUILD_SHARED "Build libhipsplit_c, a shared library exposing the C ABI for FFI" ON)
option(HIPSPLIT_WITH_MPI     "Build the optional MPI cluster-topology discovery helper" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hipsplit_core STATIC src/planner.cpp)
target_include_directories(hipsplit_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Built position-independent so the static core can be folded into a shared lib.
set_target_properties(hipsplit_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(hipsplit::core ALIAS hipsplit_core)

if(HIPSPLIT_WITH_HIP)
    # Where does ROCm live? Honour an explicit -DROCM_PATH, then the ROCM_PATH
    # environment variable, then the stock /opt/rocm. Keep it in the cache so a
    # -DROCM_PATH on the command line actually sticks across reconfigures.
    if(DEFINED ENV{ROCM_PATH} AND NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm installation root")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation root")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

    # Preferred path: the upstream HIP CMake package, which hands us hip::host.
    find_package(hip CONFIG QUIET)

    set(_hipsplit_hip_ok   FALSE)
    set(_hipsplit_hip_libs "")
    set(_hipsplit_hip_incs "")
    if(hip_FOUND)
        set(_hipsplit_hip_ok   TRUE)
        set(_hipsplit_hip_libs hip::host)
        message(STATUS "hipsplit: using the HIP CMake package from ${ROCM_PATH}")
    else()
        # Fallback: the device-query / occupancy / executor code is host-only, so
        # all we actually need is the HIP header and the runtime shared library.
        # This keeps the layer buildable on installs that ship the runtime
        # without the full CMake package (e.g. a runtime-only ROCm).
        find_path(HIPSPLIT_HIP_INCLUDE_DIR hip/hip_runtime.h
                  HINTS "${ROCM_PATH}/include")
        find_library(HIPSPLIT_HIP_RUNTIME amdhip64
                     HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64")
        if(HIPSPLIT_HIP_INCLUDE_DIR AND HIPSPLIT_HIP_RUNTIME)
            set(_hipsplit_hip_ok   TRUE)
            set(_hipsplit_hip_libs "${HIPSPLIT_HIP_RUNTIME}")
            set(_hipsplit_hip_incs "${HIPSPLIT_HIP_INCLUDE_DIR}")
            message(STATUS "hipsplit: using HIP runtime ${HIPSPLIT_HIP_RUNTIME}")
        endif()
    endif()

    if(_hipsplit_hip_ok)
        add_library(hipsplit_hip STATIC
            src/device_query.cpp
            src/occupancy.cpp
            src/autoplan.cpp
            src/executor.cpp
            src/hipsplit_c.cpp)
        target_include_directories(hipsplit_hip PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${_hipsplit_hip_incs})
        target_link_libraries(hipsplit_hip PUBLIC hipsplit_core ${_hipsplit_hip_libs})
        # Needed so a non-hipcc compiler (plain g++) can include hip_runtime.h.
        target_compile_definitions(hipsplit_hip PUBLIC __HIP_PLATFORM_AMD__)
        # PIC so this static layer can be linked into the shared C-ABI library.
        set_target_properties(hipsplit_hip PROPERTIES POSITION_INDEPENDENT_CODE ON)
        add_library(hipsplit::hip ALIAS hipsplit_hip)

        add_executable(hipsplit_print_topology examples/print_topology.cpp)
        target_link_libraries(hipsplit_print_topology PRIVATE hipsplit_hip)

        # The shared library is the artefact other languages load over FFI. It
        # carries the C ABI and the whole planner + HIP layer inside one .so,
        # exporting only the documented C entry points.
        if(HIPSPLIT_BUILD_SHARED)
            add_library(hipsplit_c SHARED src/hipsplit_c.cpp)
            target_include_directories(hipsplit_c PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include ${_hipsplit_hip_incs})
            target_link_libraries(hipsplit_c PRIVATE hipsplit_hip)
            target_compile_definitions(hipsplit_c PRIVATE HIPSPLIT_BUILD_SHARED)
            set_target_properties(hipsplit_c PROPERTIES
                OUTPUT_NAME hipsplit_c
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
                VERSION ${PROJECT_VERSION}
                SOVERSION ${PROJECT_VERSION_MAJOR})
            add_library(hipsplit::c ALIAS hipsplit_c)
        endif()

        # Optional MPI cluster-discovery helper. Off by default and built in its
        # own target so MPI never touches the core or the default build. Needs
        # the HIP layer (it queries the local GPUs) plus an MPI implementation.
        if(HIPSPLIT_WITH_MPI)
            find_package(MPI QUIET COMPONENTS CXX)
            if(MPI_FOUND)
                add_library(hipsplit_mpi STATIC src/cluster_mpi.cpp)
                target_include_directories(hipsplit_mpi PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
                target_link_libraries(hipsplit_mpi PUBLIC hipsplit_hip MPI::MPI_CXX)
                set_target_properties(hipsplit_mpi PROPERTIES POSITION_INDEPENDENT_CODE ON)
                add_library(hipsplit::mpi ALIAS hipsplit_mpi)
                message(STATUS "hipsplit: building the optional MPI cluster-discovery helper")
            else()
                message(STATUS "hipsplit: HIPSPLIT_WITH_MPI is on but no MPI was found; "
                               "skipping the cluster-discovery helper.")
            endif()
        endif()
    else()
        message(STATUS "hipsplit: no HIP install found under '${ROCM_PATH}'. "
                       "Building planning core only. If ROCm lives elsewhere, or "
                       "the dev headers aren't installed, pass -DROCM_PATH=/path "
                       "(it needs hip/hip_runtime.h and libamdhip64).")
    endif()
endif()

if(HIPSPLIT_BUILD_TESTS)
    enable_testing()

    # The planner tests are pure C++ and run on any machine, GPU or not.
    add_executable(test_planner tests/test_planner.cpp)
    target_include_directories(test_planner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_planner PRIVATE hipsplit_core)
    add_test(NAME planner COMMAND test_planner)

    # The executor smoke test needs the HIP layer, so it is only built when that
    # layer exists. It still skips at runtime when no GPU is visible, so it is
    # safe to register unconditionally here.
    if(TARGET hipsplit_hip)
        add_executable(test_executor tests/test_executor.cpp)
        target_include_directories(test_executor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(test_executor PRIVATE hipsplit_hip)
        add_test(NAME executor COMMAND test_executor)
    endif()
endif()
