#pragma once

#include <cstdint>

namespace rng {

inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline uint64_t mix(uint64_t a, uint64_t b) {
    uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

inline int8_t ternary_from_u64(uint64_t x) {
    uint8_t r = static_cast<uint8_t>(x & 0xFF);
    if (r < 85) return -1;
    if (r < 170) return 0;
    return 1;
}

inline int8_t ternary_hash(uint64_t seed, uint64_t idx, int anti_sign) {
    uint64_t s = seed;
    uint64_t v = mix(s, idx);
    return static_cast<int8_t>(ternary_from_u64(v) * anti_sign);
}

} // namespace rng

