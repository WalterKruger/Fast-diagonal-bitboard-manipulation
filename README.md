# Purpose
Fast methods I came up with for performing bit manipulation on the "diagonals" of a 8x8 bit table/bitboard, when represented using a 64-bit integer.
Two operations are included:
- A "diagonal shift", where each byte is bit shifted left/right one more/less than the previous.
- Extracting the "diagonal bits" so they are placed horizontally next to each other.
- Depositing the bits from a u8 into the bytes from the bitboard (equivalent to undoing a diagonal or vertical extract) 

Could be useful for chess programming when using a classic bitboard.
E.g. Quickly calculating non-blocked moves for a bishop when used in conjunction with count leading/trailing zeros.

# Performance
Time taken to calculate 1 billion results. Input was from an array of random valued 64-bit ints (n=20k).

'Perf' is in seconds. On my `Ryzen 5 5500` using GCC at `-O3`, `-march=native` or the default.
### Diagonal shift
| Method | Perf <sub>(m=n)</sub> | Perf <sub>(m=x86-64)</sub> |
| - | - | - |
| Linear | 3.46 | 3.38 |
| "Binary" | 1.46 | 1.47 |
| SSE left | 0.62 | 0.62 |
| SSE right | 0.73 | 0.62 |

### Extract diagonal
*The `pext` method uses a single BMI2 intrinsic. Not available on older CPUs.*
| Method | Perf <sub>(m=n)</sub> | Perf <sub>(m=x86-64)</sub> |
| - | - | - |
| [Pseudo rot45](https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#Pseudo-Rotation_by_45_degrees) | 1.20 | 1.22 |
| Psudo rot45 B | 1.10 | 1.12 |
| SSE | 0.64 | 0.73 |
| pext | 0.77 | N/A |

### Deposite uint8_t into bitboard
*The `pdep` method uses a single BMI2 intrinsic. Not available on older CPUs.*
| Method | Perf <sub>(m=n)</sub> | Perf <sub>(m=x86-64)</sub> |
| - | - | - |
| Vertical "Binary" | 1.41 | 1.42 |
| Vertical Mul. | 0.89 | 0.89 |
| Vertical clMul. | 0.95 | N/A |
| Vertical pdep | 0.81 | N/A |
| Vertical SSE | 0.73 | 0.75 |
| Diag. back Mul. | 0.75 | 0.75 |
| Diag. back SSE | 0.56 | 0.60 |
| Diag. foward Mul. | 1.23 | 1.22 |
| Diag. back SSE | 0.73 | 0.81 |
| Diag. back pdep | 0.80 | N/A |


## Diagonal shift
*File: `diagShift.h`*

For example, on a 3x3 table, a bottom-to-left diagonal shift would look like:
```
ABC'ABC'ABC => ABC'BC_'C__

ABC    ABC
ABC => BC_
ABC    C__
```
### "Binary" method
*`Functions with the '_bin' suffix`*

Works similarly to the "[binary search" based method](https://en.wikipedia.org/wiki/Find_first_set#CTZ) when calculating the number of trailing zeros.
1. Bytes that are unaffected are saved using a mask
2. For shifted bytes, only the bits came from that byte are saved
3. The two parts are combined using a bitwise OR 
```
	Start	<< 2	<< 1
	ABCD  	ABCD  	ABCD
	ABCD  	ABCD  	BCD- |
	ABCD  	CD-- |	CD00
	ABCD  	CD-- |	D00- |
```
### SSE2 method
*`Functions with the '_SSE' suffix`*

Work because each row is 8-bits wide so they can be operated on individually using special SIMD instructions.
Only uses SSE2 instructions so it can work on any x86 64-bit CPU.

Unfortunately, there is no 8-bit shift or multiply instruction, nor is there a "shift by value in corresponding vector" until AVX2, so we must do some packing/unpacking
(`_mm_sll_epi16` shifts **every** element in the first vector by the value in the lower 64-bits of the second).
1. Load into 128-bit vector and interleave with zero. Effectively zero extends 8-bit to 16-bit.
2. Multiply the 16-bit element with powers of two, which acts like a shift.
3. Un-interleave by zeroing out the upper 8-bits and convert its 16-bit elements to 8-bit using `_mm_packus_epi16` saturation conversion.
4. Return the lower 64-bits

If it needs to shift right, it instead interleaves the lower 8-bits with zero and takes the high result of the multiplication (the upper 16-bits of the intermediate 32-bit result).
When multiplying by $`1<<(8-i)`$, it effectively shifts right by $`i`$ as shifting left by 8 shifts the entire byte into the high result of the multiplication.

## Extract diagonal bits
This works very similar to the diagonal shift. The binary method doesn't need to mask on any alignment so it can directly shift the diagonals in the lower 8-bits.

The SSE2 based methods calculates a diagonal shift-to-the-left, and then extract the most significant bits of each 8-bit element using `_mm_movemask_epi8`.
