# include <stdio.h>
# include <stdint.h>
# include <emmintrin.h> // SSE2
# include <stdlib.h>    // Create random array
# include <time.h>      // Performance messuring
# include <immintrin.h> // Rotation shift

# include "diagShift.h"
# include "diagToHorizontal.h"

#define hex2d_as_u64(r7, r6, r5, r4, r3, r2, r1, r0) (0x ## r7 ## r6 ## r5 ## r4 ## r3 ## r2 ## r1 ## r0 ## ULL)

void printSSE_16(const __m128i vecToPrint) {
    union {__m128i vec; uint16_t el[8];} parts;
    parts.vec = vecToPrint;

    printf("[");
    for (size_t i=0; i<8; i++)
        printf("%4X, ", parts.el[i]);
    
    printf("\b\b]\n");
}


uint64_t antiClock_rot45(uint64_t toRotate) {
#if 0
    // https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#Pseudo-Rotation_by_45_degrees
    const uint64_t k1 = 0xAAAAAAAAAAAAAAAALLU;
    const uint64_t k2 = 0xCCCCCCCCCCCCCCCCLLU;
    const uint64_t k4 = 0xF0F0F0F0F0F0F0F0LLU;

    toRotate ^= k1 & (toRotate ^ _lrotr(toRotate,  8));
    toRotate ^= k2 & (toRotate ^ _lrotr(toRotate, 16));
    toRotate ^= k4 & (toRotate ^ _lrotr(toRotate, 32));
    return toRotate;
#else
    // "Binary" method
    uint64_t result, keptSame, changed;

    keptSame = toRotate &            0xE1E1E1E1E1E1E1E1ULL;
    changed = _rotl(toRotate, 32) & ~0xE1E1E1E1E1E1E1E1ULL;
    result = keptSame | changed;

    keptSame = result &            0x9999999999999999ULL;
    changed = _rotl(result, 16) & ~0x9999999999999999ULL;
    result = keptSame | changed;

    keptSame = result &           0x5555555555555555ULL;
    changed = _rotl(result, 8) & ~0x5555555555555555ULL;
    result = keptSame | changed;

    return result;
#endif
}



// Not a square (8x4)
#define hex2d_as_u32(r3, r2, r1, r0) (0x ## r3 ## r2 ## r1 ## r0 ## U)
uint32_t diagShift32_bin(const uint32_t toShift) {
    uint32_t result, keptSame, changed;

    keptSame = toShift &        hex2d_as_u32(F, F, 0, 0);
    changed = (toShift << 2) &  hex2d_as_u32(0, 0, C, C);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u32(F, 0, F, 0);
    changed = (result << 1) &   hex2d_as_u32(0, E, 0, E);
    result = keptSame | changed;
    
    return result;
}


int main() {
    const uint64_t rising =  0x8040201008040201ULL;
    const uint64_t falling = 0x0102040810204080ULL;

    //printf("%llX\n", antiClock_rot45(rising));
    printf("%llX\n", diagToHorizontal_fwd_SSE(falling));
    //printf("%llX\n", diagShift_tr_SSE(UINT64_MAX));


    // Performance messuring
    const size_t TOTAL_INTS = 20000;

    uint64_t rand_ints[TOTAL_INTS];
    for (size_t i=0; i < TOTAL_INTS; i++)
        rand_ints[i] = ((uint64_t)rand() << 32) | rand();


    volatile uint64_t result = 0;
    clock_t start_t = clock();

    for (size_t i=0; i< 1000000000LL; i++)
        result = diagToHorizontal_back_SSE(rand_ints[i % TOTAL_INTS]);

    clock_t end_t = clock();

    printf("\t%llu\nSeconds to finish: %.2f\n", result, (float)(end_t-start_t) / CLOCKS_PER_SEC);
}