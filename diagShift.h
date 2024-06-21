#ifndef DIAG_SHIFT_H
#define DIAG_SHIFT_H

#include <stdint.h>
#include <limits.h>     // For masking
#include <emmintrin.h>  // SIMD (SSE2)

#define hex2d_as_u64(r7, r6, r5, r4, r3, r2, r1, r0) (0x ## r7 ## r6 ## r5 ## r4 ## r3 ## r2 ## r1 ## r0 ## ULL)

// ==================
//       Linear
// ==================
// Bottom to the left (\ -> |)
uint64_t diagShift_bl_lin(uint64_t toShift) {
    union {uint64_t full; uint8_t byte[8];} inp;
    inp.full = toShift;

    for (size_t i=0; i < sizeof(uint64_t)-1; i++)
        inp.byte[i] <<= 7 - i;
    
    return inp.full;
}


// ==================
//  "Binary" method
// ==================

/*  Start       : << 4      : << 2      : << 1
	ABCDEFGH    ABCDEFGH    ABCDEFGH    ABCDEFGH
	ABCDEFGH    ABCDEFGH    ABCDEFGH    BCDEFGH- |
	ABCDEFGH    ABCDEFGH    CDEFGH-- |  CDEFGH00
	ABCDEFGH    ABCDEFGH    CDEFGH-- |  DEFGH00- |
	ABCDEFGH    EFGH---- |  EFGH0000    EFGH0000
	ABCDEFGH    EFGH---- |  EFGH0000    FGH0000- |
	ABCDEFGH    EFGH---- |  GH0000-- |  GH000000
	ABCDEFGH    EFGH---- |  GH0000-- |  H000000- |
*/
// Bottom to the left (\ -> |)
uint64_t diagShift_bl_bin(const uint64_t toShift) {
    uint64_t result, keptSame, changed;

    keptSame = toShift &        hex2d_as_u64(FF, FF, FF, FF, 00, 00, 00, 00);
    changed = (toShift << 4) &  hex2d_as_u64(00, 00, 00, 00, F0, F0, F0, F0);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(FF, FF, 00, 00, FF, FF, 00, 00);
    changed = (result << 2) &   hex2d_as_u64(00, 00, FC, FC, 00, 00, FC, FC);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(FF, 00, FF, 00, FF, 00, FF, 00);
    changed = (result << 1) &   hex2d_as_u64(00, FE, 00, FE, 00, FE, 00, FE);
    result = keptSame | changed;
    
    return result;
}

// Top to the left (/ -> |)
uint64_t diagShift_tl_bin(const uint64_t toShift) {
    uint64_t result, keptSame, changed;

    keptSame = toShift &        hex2d_as_u64(00, 00, 00, 00, FF, FF, FF, FF);
    changed = (toShift << 4) &  hex2d_as_u64(F0, F0, F0, F0, 00, 00, 00, 00);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(00, 00, FF, FF, 00, 00, FF, FF);
    changed = (result << 2) &   hex2d_as_u64(FC, FC, 00, 00, FC, FC, 00, 00);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(00, FF, 00, FF, 00, FF, 00, FF);
    changed = (result << 1) &   hex2d_as_u64(FE, 00, FE, 00, FE, 00, FE, 00);
    result = keptSame | changed;
    
    return result;
}


// Bottom to the right (/ -> |)
uint64_t diagShift_br_bin(const uint64_t toShift) {
    uint64_t result, keptSame, changed;

    keptSame = toShift &        hex2d_as_u64(FF, FF, FF, FF, 00, 00, 00, 00);
    changed = (toShift >> 4) &  hex2d_as_u64(00, 00, 00, 00, 0F, 0F, 0F, 0F);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(FF, FF, 00, 00, FF, FF, 00, 00);
    changed = (result >> 2) &   hex2d_as_u64(00, 00, 3F, 3F, 00, 00, 3F, 3F);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(FF, 00, FF, 00, FF, 00, FF, 00);
    changed = (result >> 1) &   hex2d_as_u64(00, 7F, 00, 7F, 00, 7F, 00, 7F);
    result = keptSame | changed;
    
    return result;
}

// Top to the right (\ -> |)
uint64_t diagShift_tr_bin(const uint64_t toShift) {
    uint64_t result, keptSame, changed;

    keptSame = toShift &        hex2d_as_u64(00, 00, 00, 00, FF, FF, FF, FF);
    changed = (toShift >> 4) &  hex2d_as_u64(0F, 0F, 0F, 0F, 00, 00, 00, 00);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(00, 00, FF, FF, 00, 00, FF, FF);
    changed = (result >> 2) &   hex2d_as_u64(3F, 3F, 00, 00, 3F, 3F, 00, 00);
    result = keptSame | changed;

    keptSame = result &         hex2d_as_u64(00, FF, 00, FF, 00, FF, 00, FF);
    changed = (result >> 1) &   hex2d_as_u64(7F, 00, 7F, 00, 7F, 00, 7F, 00);
    result = keptSame | changed;
    
    return result;
}


// ==================
//     SSE based
// ==================
// Bottom to the left (\ -> |)
uint64_t diagShift_bl_SSE(const uint64_t toShift) {
    __m128i toShift_inLo = _mm_cvtsi64_si128(toShift);

    // [B,A] => [0,B,0,A]
    // (No 8-bit SIMD multiply, so convert to 16-bit)
    __m128i interleaved = _mm_unpacklo_epi8(toShift_inLo, _mm_setzero_si128()); // 4.1: _mm_cvtepu8_epi16
    __m128i interShifted = _mm_mullo_epi16(interleaved, _mm_set_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7));
    
    // 'packus' is saturation conversion, so any upper bits will fill output
    __m128i shiftedLower = _mm_and_si128(interShifted, _mm_set1_epi16(UINT8_MAX));
    __m128i unInterleaved = _mm_packus_epi16(shiftedLower, shiftedLower);

    return _mm_cvtsi128_si64(unInterleaved);
}

// Top to the left: / -> |
uint64_t diagShift_tl_SSE(const uint64_t toShift) {
    __m128i toShift_inLo = _mm_cvtsi64_si128(toShift);

    // [B,A] => [0,B,0,A]
    // (No 8-bit SIMD multiply, so convert to 16-bit)
    __m128i interleaved = _mm_unpacklo_epi8(toShift_inLo, _mm_setzero_si128()); // 4.1: _mm_cvtepu8_epi16
    __m128i interShifted = _mm_mullo_epi16(interleaved, _mm_set_epi16(1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1, 1<<0));
    
    // 'packus' is saturation conversion, so any upper bits will fill output
    __m128i shiftedLower = _mm_and_si128(interShifted, _mm_set1_epi16(UINT8_MAX));
    __m128i unInterleaved = _mm_packus_epi16(shiftedLower, shiftedLower);

    return _mm_cvtsi128_si64(unInterleaved);
}

// Bottom to the right: / -> |
uint64_t diagShift_br_SSE(const uint64_t toShift) {
    __m128i toShift_inLo = _mm_cvtsi64_si128(toShift);

    // [B,A] => [B,0,A,0]
    // (No 8-bit SIMD multiply, so convert to 16-bit)
    __m128i interleaved = _mm_unpacklo_epi8(_mm_setzero_si128(), toShift_inLo);
    
    // Multiply by "1<<8" shifts into low 8 of high. Lesser power acts like a right shift
    __m128i interShifted = _mm_mulhi_epu16(interleaved, _mm_set_epi16(1<<8, 1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1));
    __m128i unInterleaved = _mm_packus_epi16(interShifted, interShifted);

    return _mm_cvtsi128_si64(unInterleaved);
}

// Top to the right: \ -> |
uint64_t diagShift_tr_SSE(const uint64_t toShift) {
    __m128i toShift_inLo = _mm_cvtsi64_si128(toShift);

    // [B,A] => [B,0,A,0]
    // (No 8-bit SIMD multiply, so convert to 16-bit)
    __m128i interleaved = _mm_unpacklo_epi8(_mm_setzero_si128(), toShift_inLo);
    
    // Multiply by "1<<8" shifts into low 8 of high. Lesser power acts like a right shift
    __m128i interShifted = _mm_mulhi_epu16(interleaved, _mm_set_epi16(1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8));
    __m128i unInterleaved = _mm_packus_epi16(interShifted, interShifted);

    return _mm_cvtsi128_si64(unInterleaved);
}


#endif