#pragma once
// arm_neon.h — x86-64 shim for the cactus inference engine
//
// Provides the ARM NEON intrinsics used by cactus, implemented on top of
// SSE4.2 + AVX2 + FMA + F16C.  Only the intrinsics actually called by this
// codebase are implemented.

#include <immintrin.h>   // SSE2 … AVX2, FMA, F16C
#include <stdint.h>
#include <string.h>
#include <math.h>

// ── Scalar float16 ────────────────────────────────────────────────────────────
#ifndef __fp16
typedef _Float16 __fp16;
#endif
typedef _Float16 float16_t;

// ── 128-bit types ─────────────────────────────────────────────────────────────
// float32x4_t maps to __m128; all 128-bit integer flavours map to __m128i.
// float16x8_t uses a GCC vector-extension type so brace initialisation works:
//   float16x8_t v = {a, b, c, d, e, f, g, h};
typedef __m128  float32x4_t;
typedef _Float16 float16x8_t __attribute__((vector_size(16)));
typedef __m128i int8x16_t;
typedef __m128i uint8x16_t;
typedef __m128i int16x8_t;
typedef __m128i int32x4_t;
typedef __m128i uint32x4_t;

// ── 64-bit types (struct-wrapped) ────────────────────────────────────────────
typedef struct { _Float16  val[4]; } float16x4_t;
typedef struct { float     val[2]; } float32x2_t;
typedef struct { int8_t    val[8]; } int8x8_t;
typedef struct { int16_t   val[4]; } int16x4_t;
typedef struct { int32_t   val[2]; } int32x2_t;

// ── Multi-vector structs ──────────────────────────────────────────────────────
typedef struct { float32x4_t val[2]; } float32x4x2_t;
typedef struct { float16x8_t val[2]; } float16x8x2_t;
typedef struct { int8x16_t   val[4]; } int8x16x4_t;

// ── float16x8_t ↔ __m128i conversion helpers ─────────────────────────────────
static inline __m128i      _f16x8_vi(float16x8_t v) { return (__m128i)v; }
static inline float16x8_t  _vi_f16x8(__m128i v)     { return (float16x8_t)v; }

// ── F16C helpers ──────────────────────────────────────────────────────────────
static inline void _neon_f16x8_to_f32(float16x8_t a,
                                        float32x4_t *lo, float32x4_t *hi) {
    __m128i ai = _f16x8_vi(a);
    *lo = _mm_cvtph_ps(ai);
    *hi = _mm_cvtph_ps(_mm_srli_si128(ai, 8));
}
static inline float16x8_t _neon_f32_to_f16x8(float32x4_t lo, float32x4_t hi) {
    __m128i a16 = _mm_cvtps_ph(lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i b16 = _mm_cvtps_ph(hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _vi_f16x8(_mm_unpacklo_epi64(a16, b16));
}

// ============================================================
// float32x4_t
// ============================================================
static inline float32x4_t vdupq_n_f32(float a)             { return _mm_set1_ps(a); }
static inline float32x4_t vld1q_f32(const float *p)        { return _mm_loadu_ps(p); }
static inline void        vst1q_f32(float *p, float32x4_t a) { _mm_storeu_ps(p, a); }

static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return _mm_add_ps(a, b); }
static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return _mm_sub_ps(a, b); }
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return _mm_mul_ps(a, b); }
static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) { return _mm_div_ps(a, b); }
static inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) { return _mm_max_ps(a, b); }
static inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) { return _mm_min_ps(a, b); }
static inline float32x4_t vnegq_f32(float32x4_t a)   { return _mm_sub_ps(_mm_setzero_ps(), a); }
static inline float32x4_t vabsq_f32(float32x4_t a)   { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a); }
static inline float32x4_t vsqrtq_f32(float32x4_t a)  { return _mm_sqrt_ps(a); }
static inline float32x4_t vrndmq_f32(float32x4_t a)  { return _mm_floor_ps(a); }

static inline float32x4_t vmulq_n_f32(float32x4_t a, float n) {
    return _mm_mul_ps(a, _mm_set1_ps(n));
}
static inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
    return _mm_fmadd_ps(b, c, a);
}
static inline float32x4_t vfmaq_n_f32(float32x4_t a, float32x4_t b, float n) {
    return _mm_fmadd_ps(b, _mm_set1_ps(n), a);
}
static inline float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
    return _mm_add_ps(a, _mm_mul_ps(b, c));
}
static inline float32x4_t vmlaq_n_f32(float32x4_t a, float32x4_t b, float n) {
    return _mm_add_ps(a, _mm_mul_ps(b, _mm_set1_ps(n)));
}

static inline float32x4_t vcvtq_f32_s32(int32x4_t a)   { return _mm_cvtepi32_ps(a); }
static inline int32x4_t   vcvtq_s32_f32(float32x4_t a) { return _mm_cvttps_epi32(a); }
static inline int32x4_t   vcvtnq_s32_f32(float32x4_t a){ return _mm_cvtps_epi32(a); }

static inline uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b) {
    return _mm_castps_si128(_mm_cmpeq_ps(a, b));
}
static inline uint32x4_t vcltq_f32(float32x4_t a, float32x4_t b) {
    return _mm_castps_si128(_mm_cmplt_ps(a, b));
}
static inline float32x4_t vbslq_f32(uint32x4_t mask, float32x4_t a, float32x4_t b) {
    return _mm_blendv_ps(b, a, _mm_castsi128_ps(mask));
}

static inline float vmaxvq_f32(float32x4_t a) {
    __m128 h = _mm_max_ps(a, _mm_movehl_ps(a, a));
    h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}
static inline float vaddvq_f32(float32x4_t a) {
    __m128 h = _mm_add_ps(a, _mm_movehl_ps(a, a));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

static inline float32x4_t vreinterpretq_f32_s32(int32x4_t a)   { return _mm_castsi128_ps(a); }
static inline float32x4_t vreinterpretq_f32_u32(uint32x4_t a)  { return _mm_castsi128_ps(a); }
static inline int32x4_t   vreinterpretq_s32_f32(float32x4_t a) { return _mm_castps_si128(a); }
static inline uint32x4_t  vreinterpretq_u32_f32(float32x4_t a) { return _mm_castps_si128(a); }

static inline float32x2_t vget_low_f32(float32x4_t a) {
    float32x2_t r; _mm_storel_pi((__m64 *)r.val, a); return r;
}
static inline float32x2_t vget_high_f32(float32x4_t a) {
    float32x2_t r; _mm_storeh_pi((__m64 *)r.val, a); return r;
}
static inline float32x4_t vcombine_f32(float32x2_t lo, float32x2_t hi) {
    return _mm_loadh_pi(_mm_loadl_pi(_mm_setzero_ps(), (const __m64 *)lo.val),
                        (const __m64 *)hi.val);
}
static inline float vget_lane_f32(float32x2_t a, int lane) { return a.val[lane]; }

static inline float32x2_t vadd_f32(float32x2_t a, float32x2_t b) {
    float32x2_t r = { a.val[0] + b.val[0], a.val[1] + b.val[1] };
    return r;
}

// transpose: val[0]={a0,b0,a2,b2}  val[1]={a1,b1,a3,b3}
static inline float32x4x2_t vtrnq_f32(float32x4_t a, float32x4_t b) {
    float32x4x2_t r;
    __m128 lo = _mm_unpacklo_ps(a, b);
    __m128 hi = _mm_unpackhi_ps(a, b);
    r.val[0] = _mm_movelh_ps(lo, hi);
    r.val[1] = _mm_movehl_ps(hi, lo);
    return r;
}

// ============================================================
// float16x8_t  (GCC vector, 128-bit, 8 × _Float16)
// ============================================================
static inline float16x8_t vdupq_n_f16(__fp16 x) {
    float32x4_t f = _mm_set1_ps((float)x);
    __m128i lo = _mm_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _vi_f16x8(_mm_unpacklo_epi64(lo, lo));
}
static inline float16x8_t vld1q_f16(const __fp16 *p) {
    return _vi_f16x8(_mm_loadu_si128((const __m128i *)p));
}
static inline void vst1q_f16(__fp16 *p, float16x8_t a) {
    _mm_storeu_si128((__m128i *)p, _f16x8_vi(a));
}

// Binary arithmetic (promote → operate → demote)
#define _NEON_F16X8_BINOP(fn, op32)                              \
static inline float16x8_t fn(float16x8_t a, float16x8_t b) {   \
    float32x4_t al, ah, bl, bh;                                  \
    _neon_f16x8_to_f32(a, &al, &ah);                            \
    _neon_f16x8_to_f32(b, &bl, &bh);                            \
    return _neon_f32_to_f16x8(op32(al, bl), op32(ah, bh));      \
}
_NEON_F16X8_BINOP(vaddq_f16, _mm_add_ps)
_NEON_F16X8_BINOP(vsubq_f16, _mm_sub_ps)
_NEON_F16X8_BINOP(vmulq_f16, _mm_mul_ps)
_NEON_F16X8_BINOP(vdivq_f16, _mm_div_ps)
_NEON_F16X8_BINOP(vmaxq_f16, _mm_max_ps)
_NEON_F16X8_BINOP(vminq_f16, _mm_min_ps)
#undef _NEON_F16X8_BINOP

static inline float16x8_t vabsq_f16(float16x8_t a) {
    // Clear sign bit (bit 15) of each 16-bit element.
    return _vi_f16x8(_mm_andnot_si128(_mm_set1_epi16((int16_t)0x8000),
                                       _f16x8_vi(a)));
}
static inline float16x8_t vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
    float32x4_t al, ah, bl, bh, cl, ch;
    _neon_f16x8_to_f32(a, &al, &ah);
    _neon_f16x8_to_f32(b, &bl, &bh);
    _neon_f16x8_to_f32(c, &cl, &ch);
    return _neon_f32_to_f16x8(_mm_fmadd_ps(bl, cl, al), _mm_fmadd_ps(bh, ch, ah));
}
static inline float16x8_t vfmsq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
    float32x4_t al, ah, bl, bh, cl, ch;
    _neon_f16x8_to_f32(a, &al, &ah);
    _neon_f16x8_to_f32(b, &bl, &bh);
    _neon_f16x8_to_f32(c, &cl, &ch);
    return _neon_f32_to_f16x8(_mm_fnmadd_ps(bl, cl, al), _mm_fnmadd_ps(bh, ch, ah));
}

static inline __fp16 vgetq_lane_f16(float16x8_t a, int lane) {
    _Float16 tmp[8]; memcpy(tmp, &a, 16); return tmp[lane];
}
static inline float16x8_t vsetq_lane_f16(__fp16 v, float16x8_t a, int lane) {
    _Float16 tmp[8]; memcpy(tmp, &a, 16); tmp[lane] = v;
    float16x8_t r; memcpy(&r, tmp, 16); return r;
}

static inline float16x8_t  vreinterpretq_f16_f32(float32x4_t a) {
    return _vi_f16x8(_mm_castps_si128(a));
}
static inline float32x4_t  vreinterpretq_f32_f16(float16x8_t a) {
    return _mm_castsi128_ps(_f16x8_vi(a));
}

static inline float16x4_t vget_low_f16(float16x8_t a) {
    float16x4_t r;
    _mm_storel_epi64((__m128i *)r.val, _f16x8_vi(a));
    return r;
}
static inline float16x4_t vget_high_f16(float16x8_t a) {
    float16x4_t r;
    _mm_storel_epi64((__m128i *)r.val, _mm_srli_si128(_f16x8_vi(a), 8));
    return r;
}
static inline float16x8_t vcombine_f16(float16x4_t lo, float16x4_t hi) {
    return _vi_f16x8(_mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i *)lo.val),
        _mm_loadl_epi64((const __m128i *)hi.val)));
}

// transpose: val[0]={a0,b0,a2,b2,...}  val[1]={a1,b1,...}
static inline float16x8x2_t vtrnq_f16(float16x8_t a, float16x8_t b) {
    float16x8x2_t r;
    __m128i ai = _f16x8_vi(a), bi = _f16x8_vi(b);
    __m128i mask_lo = _mm_set1_epi32(0x0000FFFF);
    r.val[0] = _vi_f16x8(_mm_or_si128(_mm_and_si128(ai, mask_lo),
                          _mm_slli_epi32(_mm_and_si128(bi, mask_lo), 16)));
    r.val[1] = _vi_f16x8(_mm_or_si128(_mm_srli_epi32(ai, 16),
                          _mm_and_si128(bi, _mm_set1_epi32((int)0xFFFF0000u))));
    return r;
}

// Deinterleaved load: val[0]=evens, val[1]=odds
static inline float16x8x2_t vld2q_f16(const __fp16 *p) {
    __m128i a = _mm_loadu_si128((const __m128i *)p);
    __m128i b = _mm_loadu_si128((const __m128i *)(p + 8));
    __m128i shuf_even = _mm_set_epi8(
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)13,(char)12,(char)9,(char)8,
        (char)5,(char)4,(char)1,(char)0);
    __m128i shuf_odd = _mm_set_epi8(
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)15,(char)14,(char)11,(char)10,
        (char)7,(char)6,(char)3,(char)2);
    float16x8x2_t r;
    r.val[0] = _vi_f16x8(_mm_unpacklo_epi64(_mm_shuffle_epi8(a, shuf_even),
                                              _mm_shuffle_epi8(b, shuf_even)));
    r.val[1] = _vi_f16x8(_mm_unpacklo_epi64(_mm_shuffle_epi8(a, shuf_odd),
                                              _mm_shuffle_epi8(b, shuf_odd)));
    return r;
}

// Interleaved store
static inline void vst2q_f16(__fp16 *p, float16x8x2_t a) {
    _mm_storeu_si128((__m128i *)p,
        _mm_unpacklo_epi16(_f16x8_vi(a.val[0]), _f16x8_vi(a.val[1])));
    _mm_storeu_si128((__m128i *)(p + 8),
        _mm_unpackhi_epi16(_f16x8_vi(a.val[0]), _f16x8_vi(a.val[1])));
}

// ============================================================
// float16x4_t  (64-bit, 4 × _Float16, struct-wrapped)
// ============================================================
static inline float16x4_t vld1_f16(const __fp16 *p) {
    float16x4_t r; memcpy(r.val, p, 8); return r;
}
static inline void vst1_f16(__fp16 *p, float16x4_t a) { memcpy(p, a.val, 8); }
static inline __fp16 vget_lane_f16(float16x4_t a, int lane) { return a.val[lane]; }

// float16x4_t ↔ float32x4_t (F16C)
static inline float32x4_t vcvt_f32_f16(float16x4_t a) {
    return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)a.val));
}
static inline float16x4_t vcvt_f16_f32(float32x4_t a) {
    float16x4_t r;
    _mm_storel_epi64((__m128i *)r.val,
        _mm_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return r;
}

static inline float16x4_t vadd_f16(float16x4_t a, float16x4_t b) {
    return vcvt_f16_f32(_mm_add_ps(vcvt_f32_f16(a), vcvt_f32_f16(b)));
}

// vext: concatenate a||b, extract 4 elements starting at position n
static inline float16x4_t vext_f16(float16x4_t a, float16x4_t b, int n) {
    float16x4_t r;
    for (int i = 0; i < 4; i++) {
        int idx = i + n;
        r.val[i] = (idx < 4) ? a.val[idx] : b.val[idx - 4];
    }
    return r;
}

// ============================================================
// int32x4_t / uint32x4_t
// ============================================================
static inline int32x4_t  vdupq_n_s32(int32_t n)            { return _mm_set1_epi32(n); }
static inline int32x4_t  vaddq_s32(int32x4_t a, int32x4_t b) { return _mm_add_epi32(a, b); }
static inline int32x4_t  vsubq_s32(int32x4_t a, int32x4_t b) { return _mm_sub_epi32(a, b); }
static inline int32x4_t  vandq_s32(int32x4_t a, int32x4_t b)   { return _mm_and_si128(a, b); }
static inline uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b) { return _mm_and_si128(a, b); }
static inline uint32x4_t vorrq_u32(uint32x4_t a, uint32x4_t b) { return _mm_or_si128(a, b); }
static inline int32x4_t  vshlq_n_s32(int32x4_t a, int n)    { return _mm_slli_epi32(a, n); }

// reinterpret (all 128-bit int types share __m128i)
static inline int32x4_t   vreinterpretq_s32_s8(int8x16_t a)    { return a; }
static inline int32x4_t   vreinterpretq_s32_u32(uint32x4_t a)  { return a; }
static inline int8x16_t   vreinterpretq_s8_s32(int32x4_t a)    { return a; }
static inline int8x16_t   vreinterpretq_s8_u8(uint8x16_t a)    { return a; }
static inline uint32x4_t  vreinterpretq_u32_s32(int32x4_t a)   { return a; }

// zip
static inline int32x4_t vzip1q_s32(int32x4_t a, int32x4_t b) { return _mm_unpacklo_epi32(a, b); }
static inline int32x4_t vzip2q_s32(int32x4_t a, int32x4_t b) { return _mm_unpackhi_epi32(a, b); }

static inline int32x2_t vget_low_s32(int32x4_t a) {
    int32x2_t r; _mm_storel_epi64((__m128i *)r.val, a); return r;
}
static inline int32x2_t vget_high_s32(int32x4_t a) {
    int32x2_t r; _mm_storel_epi64((__m128i *)r.val, _mm_srli_si128(a, 8)); return r;
}
static inline int32x4_t vcombine_s32(int32x2_t lo, int32x2_t hi) {
    return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)lo.val),
                               _mm_loadl_epi64((const __m128i *)hi.val));
}
static inline int32x2_t vdup_lane_s32(int32x2_t a, int lane) {
    int32x2_t r = { a.val[lane], a.val[lane] }; return r;
}
static inline int32x2_t vpadd_s32(int32x2_t a, int32x2_t b) {
    int32x2_t r = { a.val[0] + a.val[1], b.val[0] + b.val[1] }; return r;
}

// ============================================================
// int8x16_t / uint8x16_t
// ============================================================
static inline int8x16_t  vld1q_s8(const int8_t *p)  { return _mm_loadu_si128((const __m128i *)p); }
static inline uint8x16_t vld1q_u8(const uint8_t *p) { return _mm_loadu_si128((const __m128i *)p); }
static inline void vst1q_s8(int8_t *p, int8x16_t a) { _mm_storeu_si128((__m128i *)p, a); }

// Shift left each byte by n (n is a compile-time constant 0-7)
static inline int8x16_t vshlq_n_s8(int8x16_t a, int n) {
    __m128i shifted = _mm_slli_epi16(a, n);
    __m128i mask    = _mm_set1_epi8((int8_t)((uint8_t)(0xFFu << n)));
    return _mm_and_si128(shifted, mask);
}

// Arithmetic shift right each byte by n (n is a compile-time constant 0-7)
static inline int8x16_t vshrq_n_s8(int8x16_t a, int n) {
    // Sign-extend even bytes (at low 8 bits of each 16-bit word), shift right.
    __m128i even = _mm_srai_epi16(_mm_slli_epi16(a, 8), 8 + n);
    // Shift odd bytes (at high 8 bits of each 16-bit word) right.
    __m128i odd  = _mm_srai_epi16(a, n);
    __m128i mask = _mm_set1_epi16(0x00FF);
    return _mm_or_si128(_mm_and_si128(even, mask), _mm_andnot_si128(mask, odd));
}

static inline int8x8_t vget_low_s8(int8x16_t a) {
    int8x8_t r; _mm_storel_epi64((__m128i *)r.val, a); return r;
}
static inline int8x8_t vget_high_s8(int8x16_t a) {
    int8x8_t r; _mm_storel_epi64((__m128i *)r.val, _mm_srli_si128(a, 8)); return r;
}
static inline int8x16_t vcombine_s8(int8x8_t lo, int8x8_t hi) {
    return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)lo.val),
                               _mm_loadl_epi64((const __m128i *)hi.val));
}

// ============================================================
// int8x8_t  (64-bit, struct-wrapped)
// ============================================================
static inline int8x8_t vld1_s8(const int8_t *p) { int8x8_t r; memcpy(r.val, p, 8); return r; }
static inline void vst1_s8(int8_t *p, int8x8_t a) { memcpy(p, a.val, 8); }

static inline int32x2_t vreinterpret_s32_s8(int8x8_t a) {
    int32x2_t r; memcpy(r.val, a.val, 8); return r;
}
static inline int8x8_t vreinterpret_s8_s32(int32x2_t a) {
    int8x8_t r; memcpy(r.val, a.val, 8); return r;
}

// ============================================================
// int16x8_t
// ============================================================
// Pairwise widen-add: int16x8 → int32x4
static inline int32x4_t vpaddlq_s16(int16x8_t a) {
    return _mm_madd_epi16(a, _mm_set1_epi16(1));
}
static inline int16x4_t vget_low_s16(int16x8_t a) {
    int16x4_t r; _mm_storel_epi64((__m128i *)r.val, a); return r;
}
static inline int16x4_t vget_high_s16(int16x8_t a) {
    int16x4_t r; _mm_storel_epi64((__m128i *)r.val, _mm_srli_si128(a, 8)); return r;
}
static inline int16x8_t vcombine_s16(int16x4_t lo, int16x4_t hi) {
    return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)lo.val),
                               _mm_loadl_epi64((const __m128i *)hi.val));
}
static inline int16x4_t vqmovn_s32(int32x4_t a) {
    int16x4_t r;
    _mm_storel_epi64((__m128i *)r.val, _mm_packs_epi32(a, _mm_setzero_si128()));
    return r;
}
static inline int8x8_t vqmovn_s16(int16x8_t a) {
    int8x8_t r;
    _mm_storel_epi64((__m128i *)r.val, _mm_packs_epi16(a, _mm_setzero_si128()));
    return r;
}
static inline int16x8_t vmovl_s8(int8x8_t a) {
    return _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)a.val));
}
static inline int32x4_t vmovl_s16(int16x4_t a) {
    return _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)a.val));
}
static inline int16x8_t vmull_s8(int8x8_t a, int8x8_t b) {
    __m128i a16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)a.val));
    __m128i b16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)b.val));
    return _mm_mullo_epi16(a16, b16);
}

// ============================================================
// int8x16x4_t — 64-byte table lookup
// ============================================================
static inline int8x16x4_t vld1q_s8_x4(const int8_t *p) {
    int8x16x4_t r;
    r.val[0] = _mm_loadu_si128((const __m128i *)(p));
    r.val[1] = _mm_loadu_si128((const __m128i *)(p + 16));
    r.val[2] = _mm_loadu_si128((const __m128i *)(p + 32));
    r.val[3] = _mm_loadu_si128((const __m128i *)(p + 48));
    return r;
}

// vqtbl4q_s8: look up each byte of idx (0-63) in a 64-byte table (4 × 16).
static inline int8x16_t vqtbl4q_s8(int8x16x4_t t, uint8x16_t idx) {
    __m128i i = idx;
    // table 0: valid for idx 0-15; bit-7 marks out-of-range for PSHUFB.
    __m128i gt15 = _mm_cmpgt_epi8(i, _mm_set1_epi8(15));
    __m128i r0 = _mm_shuffle_epi8(t.val[0], _mm_or_si128(i, gt15));
    // table 1: valid for idx 16-31; adj wraps negative (bit-7) for idx<16.
    __m128i adj1 = _mm_sub_epi8(i, _mm_set1_epi8(16));
    __m128i gt31 = _mm_cmpgt_epi8(i, _mm_set1_epi8(31));
    __m128i r1 = _mm_shuffle_epi8(t.val[1], _mm_or_si128(adj1, gt31));
    // table 2: valid for idx 32-47; adj wraps to 224-255 (bit-7) for idx<32.
    __m128i adj2 = _mm_sub_epi8(i, _mm_set1_epi8(32));
    __m128i gt47 = _mm_cmpgt_epi8(i, _mm_set1_epi8(47));
    __m128i r2 = _mm_shuffle_epi8(t.val[2], _mm_or_si128(adj2, gt47));
    // table 3: valid for idx 48-63; adj wraps to 208-255 (bit-7) for idx<48.
    __m128i adj3 = _mm_sub_epi8(i, _mm_set1_epi8(48));
    __m128i r3 = _mm_shuffle_epi8(t.val[3], adj3);
    return _mm_or_si128(_mm_or_si128(r0, r1), _mm_or_si128(r2, r3));
}
