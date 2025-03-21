#pragma once

#include <stdint.h>
#include <immintrin.h> // SSE2, AVX2, AVX512bw


// https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipabouttheDiagonal
uint64_t flipDiagA1H8(uint64_t x) {
    uint64_t t;
    const uint64_t k1 = UINT64_C(0x5500550055005500);
    const uint64_t k2 = UINT64_C(0x3333000033330000);
    const uint64_t k4 = UINT64_C(0x0f0f0f0f00000000);

    t  = k4 & (x ^ (x << 28));
    x ^=       t ^ (t >> 28) ;
    t  = k2 & (x ^ (x << 14));
    x ^=       t ^ (t >> 14) ;
    t  = k1 & (x ^ (x <<  7));
    x ^=       t ^ (t >>  7) ;
    return x;
}

// Vectorized version of above
__m128i flipDiagA1H8_epi64(__m128i m) {
    const __m128i k1 = _mm_set1_epi64x(0xaa00aa00aa00aa00ull);
    const __m128i k2 = _mm_set1_epi64x(0xcccc0000cccc0000ull);
    const __m128i k4 = _mm_set1_epi64x(0xf0f0f0f00f0f0f0full);

    __m128i t =                             _mm_xor_si128(m, _mm_slli_epi64(m, 36));
    m = _mm_xor_si128(m, _mm_and_si128( k4, _mm_xor_si128(t, _mm_srli_epi64(m, 36)) )) ;

    t =                  _mm_and_si128( k2, _mm_xor_si128(m, _mm_slli_epi64(m, 18)) );
    m =                  _mm_xor_si128( m,  _mm_xor_si128(t, _mm_srli_epi64(t, 18)) );

    t =                  _mm_and_si128( k1, _mm_xor_si128(m, _mm_slli_epi64(m, 9 )) );
    m =                  _mm_xor_si128( m,  _mm_xor_si128(t, _mm_srli_epi64(t, 9 )) );
    return m;
}



uint64_t diagTranspose_sse(uint64_t x) {
    // We can transpose two rows at once, so offset lower lane
    __m128i inLowOnly = _mm_cvtsi64_si128(x);
    __m128i duplicatedHiLow = _mm_shuffle_epi32(inLowOnly, _MM_SHUFFLE(1,0,1,0));
    __m128i shifted_0 = _mm_add_epi64(duplicatedHiLow, inLowOnly); // << 1

    // Calculating every shifted position first allows better out of order execution
    __m128i shifted_2 = _mm_slli_epi64(shifted_0, 2);
    __m128i shifted_4 = _mm_slli_epi64(shifted_0, 4);
    __m128i shifted_6 = _mm_slli_epi64(shifted_0, 6);

    uint32_t hiHi = _mm_movemask_epi8(shifted_0);
    uint32_t hiLo = _mm_movemask_epi8(shifted_2);
    uint32_t loHi = _mm_movemask_epi8(shifted_4);
    uint32_t loLo = _mm_movemask_epi8(shifted_6);

    uint64_t highTranspose = (hiHi << 16) | hiLo;
    return (highTranspose << 32) | (loHi << 16) | loLo; 
}


#ifdef CPU_HAS_AVX2
uint64_t diagTranspose_avx2(uint64_t x) {
    __m256i vec = _mm256_set1_epi64x(x);
    __m256i vecOffset = _mm256_sllv_epi64(vec, _mm256_setr_epi64x(3,2,1,0));

    uint32_t upperTranspose = _mm256_movemask_epi8(vecOffset);
    uint32_t lowerTranspose = _mm256_movemask_epi8(_mm256_slli_epi64(vecOffset, 4));

    return ((uint64_t)upperTranspose << 32) | lowerTranspose;
}
#endif

#ifdef CPU_HAS_AVX512 // avx512bw
uint64_t diagTranspose_avx512(uint64_t x) {
    __m512i vec = _mm512_set1_epi64(x);
    __m512i vecOffset = _mm512_sllv_epi64(vec, _mm512_setr_epi64(7,6,5,4,3,2,1,0));

    return (uint64_t)_mm512_movepi8_mask(vecOffset);
}
#endif