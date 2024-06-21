#ifndef DIAG_TO_HORIZONTAL_H
#define DIAG_TO_HORIZONTAL_H

#include <stdint.h>
#include <limits.h>     // For masking
#include <immintrin.h> // Pext & SSE2

#define hex2d_as_u64(r7, r6, r5, r4, r3, r2, r1, r0) (0x ## r7 ## r6 ## r5 ## r4 ## r3 ## r2 ## r1 ## r0 ## ULL)

//#define CPU_HAS_BMI2


#ifdef CPU_HAS_BMI2
// Faster when returning a u8 (pext only?)
// Anti-clockwise (\ -> -)
uint8_t diagToHorizontal_back_pext(const uint64_t toShift) {
    return _pext_u64(toShift, hex2d_as_u64(80, 40, 20, 10, 08, 04, 02, 01));
}

// Clockwise (/ -> -)
uint8_t diagToHorizontal_fwd_pext(const uint64_t toShift) {
    return _pext_u64(toShift, hex2d_as_u64(01, 02, 04, 08, 10, 20, 40, 80));
}
#endif

// Anti-clockwise (\ -> -)
uint64_t diagToHorizontal_back_SSE(const uint64_t toShift) {
    __m128i toShift_inLo = _mm_cvtsi64_si128(toShift);

    // [B,A] => [0,B,0,A]
    // (No 8-bit SIMD multiply, so convert to 16-bit)
    __m128i interleaved = _mm_unpacklo_epi8(toShift_inLo, _mm_setzero_si128()); // 4.1: _mm_cvtepu8_epi16
    __m128i interShifted = _mm_mullo_epi16(interleaved, _mm_set_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7));
    
    // 'packus' is saturation conversion, so any upper bits will fill output
    __m128i shiftedLower = _mm_and_si128(interShifted, _mm_set1_epi16(UINT8_MAX));
    __m128i unInterleaved = _mm_packus_epi16(shiftedLower, _mm_setzero_si128());

    return _mm_movemask_epi8(unInterleaved);
}

// Clockwise (/ -> -)
uint64_t diagToHorizontal_fwd_SSE(const uint64_t toShift) {
    __m128i toShift_inLo = _mm_cvtsi64_si128(toShift);

    // [B,A] => [0,B,0,A]
    // (No 8-bit SIMD multiply, so convert to 16-bit)
    __m128i interleaved = _mm_unpacklo_epi8(toShift_inLo, _mm_setzero_si128()); // 4.1: _mm_cvtepu8_epi16
    __m128i interShifted = _mm_mullo_epi16(interleaved, _mm_set_epi16(1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1, 1<<0));
    
    // 'packus' is saturation conversion, so any upper bits will fill output
    __m128i shiftedLower = _mm_and_si128(interShifted, _mm_set1_epi16(UINT8_MAX));
    __m128i unInterleaved = _mm_packus_epi16(shiftedLower, _mm_setzero_si128());

    return _mm_movemask_epi8(unInterleaved);
}


#endif