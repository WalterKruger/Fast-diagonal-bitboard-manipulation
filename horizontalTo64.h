#ifndef HORIZONTAL_TO_64_H
#define HORIZONTAL_TO_64_H

#include <stdint.h>
#include <limits.h>     // For masking
#include <immintrin.h> // SSE2 & clMul

#define hex2d_as_u64(r7, r6, r5, r4, r3, r2, r1, r0) (0x ## r7 ## r6 ## r5 ## r4 ## r3 ## r2 ## r1 ## r0 ## ULL)


// Clockwise (- -> \)
uint64_t toDiag_back_mul(const uint8_t input) {
    uint64_t broadcasted = input * 0x0101010101010101ULL;
    // Keep only diagonal bits
    return broadcasted & 0x8040201008040201ULL;
}

// Clockwise (- -> \)
uint64_t toDiag_back_sse(const uint8_t input) {
    __m128i boadcasted = _mm_set1_epi8(input);
    //return _mm_cvtsi128_si64(boadcasted) & 0x8040201008040201ULL;

    __m128i diagonalMask = _mm_cvtsi64_si128(0x8040201008040201ULL);
    __m128i result = _mm_and_si128(boadcasted, diagonalMask);
    return _mm_cvtsi128_si64(result);
}


// Anti-clockwise (- -> /)
uint64_t toDiag_fwd_mul(const uint8_t input) {
    // 2+6th bit set for the mul, 6-wide, so no carry
    uint64_t inp_noCarry = input & 0b00111111;
    uint64_t bitsInDiag = inp_noCarry * ((1ULL<<7) + (1ULL<<13) + (1ULL<<19) + (1ULL<<25) + (1ULL<<31) + (1ULL<<37));

    // Restore missing 2-bits
    uint64_t missingBits = input & 0b11000000;
    bitsInDiag |= missingBits << 43;
    bitsInDiag |= missingBits << 49;

    return bitsInDiag & 0x0102040810204080ULL;
}

// Anti-clockwise (- -> /)
uint64_t toDiag_fwd_sse(const uint8_t input) {
    // A => [A,0, A,0, A,0]
    __m128i interleaved = _mm_slli_epi16(_mm_set1_epi16(input), 8);

    // Mul (1<<8) moves entire element into the high result, more/less shifts left/right 
    const __m128i TO_MUL = _mm_set_epi16(1<<1, 1<<3, 1<<5, 1<<7, 1<<9, 1<<11, 1<<13, 1<<15);
    __m128i interShifted = _mm_mulhi_epu16(interleaved, TO_MUL);

    // 'packus' is saturation conversion, so any upper bits will fill output
    __m128i shiftedLower = _mm_and_si128(interShifted, _mm_set1_epi16(UINT8_MAX));
    __m128i unInterleaved = _mm_packus_epi16(shiftedLower, shiftedLower);

    const __m128i DIAGONAL_MASK = _mm_cvtsi64_si128(0x0102040810204080ULL);
    return _mm_cvtsi128_si64(_mm_and_si128(unInterleaved, DIAGONAL_MASK));
}

#ifdef CPU_HAS_BMI2
uint64_t toDiagonal_fwd_pdep(const uint8_t input) {
    return _pdep_u64(input, 0x0102040810204080ULL);
}
#endif



// =====================
//  To LSB of each byte
// =====================
// Anti-clockwise (- -> |)
uint64_t toVertical_mul(const uint8_t input) {
    // 7th bit set for the mul, 7-wide, so no carry
    uint64_t inp_noCarry = input & 0b11111110;
    uint64_t bitsInDiag = inp_noCarry * ((1ULL<<0) + (1ULL<<7) + (1ULL<<14) + (1ULL<<21) + (1ULL<<28) + (1ULL<<35) + (1ULL<<42) + (1ULL<<49));

    // Restore LSB, upper 7 will be masked away later
    bitsInDiag |= input;

    return bitsInDiag & 0x0101010101010101ULL;
}

// Anti-clockwise (- -> |)
uint64_t toVertical_sse(const uint8_t input) {
    // A => [A,0, A,0, A,0]
    __m128i interleaved = _mm_slli_epi16(_mm_set1_epi16(input), 8);

    // Mul 1<<8 shifts entire into high result [>> i := * (1 << 8-i)]
    const __m128i TO_MUL = _mm_set_epi16(1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8);
    __m128i interShifted = _mm_mulhi_epu16(interleaved, TO_MUL);

    __m128i unInterleaved = _mm_packus_epi16(interShifted, interShifted);

    const __m128i VERTICAL_MASK = _mm_cvtsi64_si128(0x0101010101010101ULL);
    return _mm_cvtsi128_si64(_mm_and_si128(unInterleaved, VERTICAL_MASK));
}

// Anti-clockwise (- -> |)
uint64_t toVertical_bin(const uint8_t input) {
    uint64_t keptSame, changed, result = (uint64_t)input;

    keptSame = result & 0b00001111;
    changed = (result & 0b11110000) << 28;
    result = keptSame | changed;

    keptSame = result & hex2d_as_u64(00, 00, 00, 03, 00, 00, 00, 03);
    changed = (result & hex2d_as_u64(00, 00, 00, 0C, 00, 00, 00, 0C)) << 14;
    result |= keptSame | changed;

    keptSame = result & hex2d_as_u64(00, 01, 00, 01, 00, 01, 00, 01);
    changed = (result & hex2d_as_u64(00, 02, 00, 02, 00, 02, 00, 02)) << 7;
    result = keptSame | changed;

    return result;
}


#ifdef CPU_HAS_BMI2
// Anti-clockwise (- -> |)
uint64_t toVertical_clMul(const uint8_t input) {
    __m128i input_vec = _mm_cvtsi32_si128(input);
    __m128i toMul = _mm_cvtsi64_si128((1ULL<<0) + (1ULL<<7) + (1ULL<<14) + (1ULL<<21) + (1ULL<<28) + (1ULL<<35) + (1ULL<<42) + (1ULL<<49));

    __m128i bitsInDiag = _mm_clmulepi64_si128(input_vec, toMul, 0x00);
    return _mm_cvtsi128_si64(bitsInDiag) & 0x0101010101010101ULL;
}

// Anti-clockwise (- -> |)
uint64_t toVertical_pdep(const uint8_t input) {
    return _pdep_u64(input, 0x0101010101010101ULL);
}
#endif

#endif