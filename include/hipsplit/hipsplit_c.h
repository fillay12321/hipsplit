#ifndef HIPSPLIT_C_H
#define HIPSPLIT_C_H

/*
 * C ABI for hipsplit.
 *
 * This exists so callers that aren't C++ can use the adaptive splitter without
 * dragging in the C++ types. A C ABI is the lingua franca of FFI: anything that
 * can call a C function — Rust, Python (ctypes/cffi), Go (cgo), Node, Julia,
 * Swift, plain C — can call this. It mirrors the planner's value structs
 * field-for-field and exposes a single planning entry point.
 *
 * For link-time consumers (C/C++/Rust) link the static library. For everyone
 * else, build the shared library (HIPSPLIT_BUILD_SHARED) and load libhipsplit_c
 * at runtime; the three functions below are the entire exported surface.
 *
 * Nothing is allocated across the boundary. The caller owns the output buffers:
 * a small launches[] array (one entry per GPU that gets work — in practice a
 * handful at most) and a char buffer for the human-readable note. The call
 * returns how many launches the plan actually has, so the caller can tell if it
 * needs a bigger buffer, though it almost never will.
 */

#include <stddef.h>
#include <stdint.h>

/*
 * Export/visibility marker for the shared-library build. When the library is
 * compiled with -DHIPSPLIT_BUILD_SHARED these symbols are exported (and all
 * other symbols stay hidden); a consumer header include needs nothing. On a
 * static build the macro is empty.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HIPSPLIT_BUILD_SHARED)
#    define HIPSPLIT_API __declspec(dllexport)
#  elif defined(HIPSPLIT_USE_SHARED)
#    define HIPSPLIT_API __declspec(dllimport)
#  else
#    define HIPSPLIT_API
#  endif
#else
#  if defined(HIPSPLIT_BUILD_SHARED)
#    define HIPSPLIT_API __attribute__((visibility("default")))
#  else
#    define HIPSPLIT_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors hipsplit::KernelProfile. Leave threads_per_block at 0 for auto. */
typedef struct hipsplit_kernel_profile {
    int    threads_per_block;
    size_t dynamic_lds_bytes;
    size_t static_lds_bytes;
    int    block_size_limit;
    int    block_size_multiple;
} hipsplit_kernel_profile;

/* Mirrors hipsplit::Workload. */
typedef struct hipsplit_workload {
    uint64_t total_items;
    int      items_per_thread;
    uint64_t split_granularity;
} hipsplit_workload;

/* Mirrors hipsplit::DeviceLaunch — what to launch on one device. */
typedef struct hipsplit_device_launch {
    int      device_id;
    uint64_t item_offset;
    uint64_t item_count;
    int      block_size;
    uint64_t grid_size;
    int      blocks_per_cu;
    double   saturation;
} hipsplit_device_launch;

/*
 * Build a profile/workload pre-filled with the same defaults the C++ structs
 * use (auto block size, 1024 block limit, items_per_thread = 1, granularity 1).
 * Start from these and overwrite the few fields you care about, so you never
 * accidentally plan against a zero-initialised workload.
 */
HIPSPLIT_API hipsplit_kernel_profile hipsplit_kernel_profile_default(void);
HIPSPLIT_API hipsplit_workload       hipsplit_workload_default(void);

/*
 * Query the HIP devices present right now and plan a launch for them.
 *
 *   kp, w         describe the kernel and the work. Pass NULL to use defaults.
 *   out_launches  caller-owned array, written with up to launch_cap entries.
 *   launch_cap    capacity of out_launches (entries, not bytes).
 *   out_valid     set to 1 if the plan is usable, 0 if not (e.g. no GPU).
 *                 May be NULL if you don't care.
 *   out_note      caller-owned buffer for the plan's note, NUL-terminated and
 *                 truncated to note_cap. May be NULL.
 *   note_cap      capacity of out_note in bytes.
 *
 * Returns the total number of launches in the plan. This may be larger than
 * launch_cap, in which case out_launches holds the first launch_cap of them and
 * the caller knows to retry with a bigger buffer. A return of 0 with
 * *out_valid == 0 means no device was available.
 *
 * Note: this PLANS, it does not launch. You still issue the kernel yourself for
 * each returned slice. See the README for the full picture.
 */
HIPSPLIT_API int hipsplit_plan_auto(const hipsplit_kernel_profile* kp,
                                    const hipsplit_workload*       w,
                                    hipsplit_device_launch*        out_launches,
                                    int                            launch_cap,
                                    int*                           out_valid,
                                    char*                          out_note,
                                    int                            note_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HIPSPLIT_C_H */
