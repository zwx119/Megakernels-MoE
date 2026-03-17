#pragma once

/**
 * ============================================================================
 *   编译兼容性补丁
 * ============================================================================
 *
 * 【问题】
 *   ThunderKittens 的 base_types.cuh 使用了 std::bit_cast（C++20，需要 GCC 11+）。
 *   如果你的 GCC 版本较旧（如 GCC 9/10），编译会报错：
 *     "namespace "std" has no member "bit_cast""
 *
 * 【解决方案】
 *   本文件提供 std::bit_cast 的兼容实现（基于 memcpy），
 *   通过 Makefile 中的 -include compat.cuh 在所有头文件之前注入。
 *
 * 【使用方法】
 *   在 Makefile 的 NVCCFLAGS 中添加:
 *     -include $(shell pwd)/compat.cuh
 *   或者在 main.cu 的最顶部 #include "compat.cuh"（在 kittens.cuh 之前）
 */

#include <cstring>
#include <type_traits>

// 如果编译器不提供 std::bit_cast，用 memcpy 实现
#if !__has_include(<bit>) || (__GNUC__ && __GNUC__ < 11)

namespace std {

template <typename To, typename From>
#if defined(__CUDA_ARCH__)
__device__
#endif
__host__
inline constexpr
typename std::enable_if<
    sizeof(To) == sizeof(From) &&
    std::is_trivially_copyable<From>::value &&
    std::is_trivially_copyable<To>::value,
    To>::type
bit_cast(const From &src) noexcept {
    To dst;
    memcpy(&dst, &src, sizeof(To));
    return dst;
}

} // namespace std

#endif // !std::bit_cast

// 如果 CUDA 版本不支持 cudaLaunchAttributePreferredClusterDimension（需要 CUDA 12.3+），
// 提供一个兼容定义。该 attribute 仅在 cluster launch 时使用，单 SM kernel 不影响。
//
// 用 #define 宏注入，在 ThunderKittens util.cuh 引用该符号之前生效。
// 由于该符号是 enum 值而非宏，无法用 #ifndef 检测，这里直接无条件定义。
// 如果你的 CUDA >= 12.3 已有此符号，请删除下面这行。
#define cudaLaunchAttributePreferredClusterDimension ((cudaLaunchAttributeID)9)

