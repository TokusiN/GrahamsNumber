#include "BigInt.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace {
	int memsize;
	double* WR;
	double* WI;
	unsigned short bitflip[65536];
}

void FFTInit(int size);
void CalcW();
void InitBitFlip();
__m256d* FFT(const CBigInt& in, int fftsize);
// __m256d* CompFFT(const CBigInt& re, const CBigInt& im, int fftsize);
void IFFT(__m256d* in, int fftsize);
void BitFlip(__m256d* in, int size);

void FFTInit(int size)
{
	memsize = NextPow2(size-1);
	CalcW();
	InitBitFlip();
}

void CalcW()
{
	int size = memsize * 2;
	if (WR)
	{
		MM_FREE(WR);
		MM_FREE(WI);
	}
	WR = MM_MALLOC(double, size);
	WI = MM_MALLOC(double, size);
	WR[0] = 1;
	WI[0] = 0;
	WR[1] = 0;
	WI[1] = 1;
	int b;
	int i;
	for (b = 2; b < size; b *= 2)
	{
		double a = double(3.141592653589793238462643383279 / (b * 2));
		double re = cos(a);
		double im = sin(a);
		for (i = 0; i < b; i++)
		{
			WR[i + b] = WR[i] * re - WI[i] * im;
			WI[i + b] = WR[i] * im + WI[i] * re;
		}
	}
}

void InitBitFlip()
{
	int h, l;
	int i;
	for (h = 1, l = 32768; h <= 32768; h *= 2, l /= 2)
	{
		for (i = 0; i < h; i++)
		{
			bitflip[i + h] = bitflip[i] + l;
		}
	}
}
/*
#include <time.h>
int ffttimes;
int ffttimel;

int fft8;
int fft16;
int fft64;
int fft256;
*/
CBigInt FFTMul(const CBigInt& a, const CBigInt& b)
{
//	int start = clock();
	int i;
	int fftsize = NextPow2((a.m_digitnum + b.m_digitnum -1) / 4);
	if (fftsize < 8)
	{
		fftsize = 8;
	}
	/*
	bool c = false;
	if (fftsize == 8)
	{
		c = true;
		fft8++;
	}
	if (fftsize == 16)
	{
		c = true;
		fft16++;
	}
	if (fftsize == 64)
	{
		c = true;
		fft64++;
	}
	if (fftsize == 256)
	{
		c = true;
		fft256++;
	}
	*/
	if (fftsize > memsize)
	{
		if (memsize == 0)
		{
			InitBitFlip();
		}
		memsize = fftsize;
		CalcW();
	}
	__m256d* ffta = FFT(a, fftsize);
	if (&a == &b)
	{
		for (i = 0; i < fftsize; i++)
		{
			__m256d ar = ffta[i * 2];
			__m256d ai = ffta[i * 2 + 1];
			__m256d re = _mm256_mul_pd(ar, ar);
			re = _mm256_fnmadd_pd(ai, ai, re);
			__m256d im = _mm256_mul_pd(ar, ai);
			im = _mm256_add_pd(im, im);
			ffta[i * 2 + 0] = re;
			ffta[i * 2 + 1] = im;
		}
	}
	else
	{
		__m256d* fftb = FFT(b, fftsize);
		for (i = 0; i < fftsize; i++)
		{
			__m256d ar = ffta[i * 2];
			__m256d ai = ffta[i * 2 + 1];
			__m256d br0 = fftb[i * 2];
			__m256d bi = fftb[i * 2 + 1];
			__m256d bbr = _mm256_mul_pd(br0, ar);
			__m256d br = _mm256_fnmadd_pd(bi, ai, bbr);
			bi = _mm256_mul_pd(bi, ar);
			bi = _mm256_fmadd_pd(br0, ai, bi);
			ffta[i * 2 + 0] = br;
			ffta[i * 2 + 1] = bi;
		}
		MM_FREE(fftb);
	}
	i = 10;
	IFFT(ffta, fftsize);
	double dmul = 1./ (fftsize * 4);
	__m256d mul = _mm256_broadcastsd_pd(_mm_load1_pd(&dmul));
	for (i = 0; i < fftsize * 2; i++)
	{
		ffta[i] = _mm256_mul_pd(ffta[i], mul);
	}
	CBigInt ret;
	ret.m_sign = a.m_sign * b.m_sign;
	ret.m_maxdigitnum = fftsize * 4;
	ret.m_digits = MM_MALLOC(int, fftsize * 4);
	auto calcaddr = [](int addr) {
		return (addr & 3) + (addr & (~3)) * 2;
	};
	double* buf = (double*)ffta;
	long long carry = 0;
	for (i = 0; i < a.m_digitnum + b.m_digitnum; i++)
	{
		int addr = calcaddr(i);
		long long data = (long long)(buf[addr] + .5) + carry;
		if (data)
		{
			ret.m_digitnum = i+1;
		}
		ret.m_digits[i] = data % BIGINT_BASE;
		carry = data / BIGINT_BASE;
	}
	MM_FREE(ffta);
/*	if (fftsize < 64)
	{
		ffttimes += clock() - start;
	}
	else
	{
		ffttimel += clock() - start;
	}
	if (c && (fft8 + fft16 + fft64 + fft256) % 1000000 == 0)
	{
		printf("fft  8:%9d\n", fft8);
		printf("fft 16:%9d\n", fft16);
		printf("fft 64:%9d\n", fft64);
		printf("fft256:%9d\n", fft256);
		printf("times :%9d\n", ffttimes);
		printf("timel :%9d\n", ffttimel);
	}*/
	return std::move(ret);
}

#if 0
CBigInt FFTCompMul(const CBigInt& ar, const CBigInt& ai, const CBigInt& br, const CBigInt& bi, CBigInt& or , CBigInt& oi)
{
	int i;
	int fftsize = NextPow2((std::max( ar.m_digitnum, ai.m_digitnum) + std::max(br.m_digitnum, bi.m_digitnum) - 1) / 4);
	if (fftsize > memsize)
	{
		if (memsize == 0)
		{
			InitBitFlip();
		}
		memsize = fftsize;
		CalcW();
	}
	__m256d* ffta = CompFFT(ar, ai, fftsize);
	if (&ar == &br && &ai == &bi)
	{
		for (i = 0; i < fftsize; i++)
		{
			__m256d ar = ffta[i * 2];
			__m256d ai = ffta[i * 2 + 1];
			__m256d re = _mm256_mul_pd(ar, ar);
			re = _mm256_fnmadd_pd(ai, ai, re);
			__m256d im = _mm256_mul_pd(ar, ai);
			im = _mm256_add_pd(im, im);
			ffta[i * 2 + 0] = re;
			ffta[i * 2 + 1] = im;
		}
	}
	else
	{
		__m256d* fftb = CompFFT(br, bi, fftsize);
		for (i = 0; i < fftsize; i++)
		{
			__m256d ar = ffta[i * 2];
			__m256d ai = ffta[i * 2 + 1];
			__m256d br0 = fftb[i * 2];
			__m256d bi = fftb[i * 2 + 1];
			__m256d bbr = _mm256_mul_pd(br0, ar);
			__m256d br = _mm256_fnmadd_pd(bi, ai, bbr);
			bi = _mm256_mul_pd(bi, ar);
			bi = _mm256_fmadd_pd(br0, ai, bi);
			ffta[i * 2 + 0] = br;
			ffta[i * 2 + 1] = bi;
		}
		MM_FREE(fftb);
	}
	IFFT(ffta, fftsize);
	double dmul = 1. / (fftsize * 4);
	__m256d mul = _mm256_broadcastsd_pd(_mm_load1_pd(&dmul));
	for (i = 0; i < fftsize * 2; i++)
	{
		ffta[i] = _mm256_mul_pd(ffta[i], mul);
	}
	CBigInt ret;
	// Todo:•„†‚ð‰½‚Æ‚©‚µ‚æ‚¤
	ret.m_sign = ar.m_sign * br.m_sign;
	ret.m_maxdigitnum = fftsize * 4;
	ret.m_digits = MM_MALLOC(int, fftsize * 4);
	auto calcaddr = [](int addr) {
		return (addr & 3) + (addr & (~3)) * 2;
	};
	double* buf = (double*)ffta;
	long long carry = 0;
	for (i = 0; i < fftsize * 4; i++)
	{
		int addr = calcaddr(i);
		long long data = (long long)(buf[addr] + .5) + carry;
		if (data < 0)
		{
			printf("!");
		}
		if (data)
		{
			ret.m_digitnum = i + 1;
		}
		ret.m_digits[i] = data % BIGINT_BASE;
		carry = data / BIGINT_BASE;
	}
	MM_FREE(ffta);
	/*	if (fftsize < 64)
	{
	ffttimes += clock() - start;
	}
	else
	{
	ffttimel += clock() - start;
	}
	if (c && (fft8 + fft16 + fft64 + fft256) % 1000000 == 0)
	{
	printf("fft  8:%9d\n", fft8);
	printf("fft 16:%9d\n", fft16);
	printf("fft 64:%9d\n", fft64);
	printf("fft256:%9d\n", fft256);
	printf("times :%9d\n", ffttimes);
	printf("timel :%9d\n", ffttimel);
	}*/
	return std::move(ret);
}
#endif

__m256d* FFT(const CBigInt& in, int fftsize)
{
	int i, j;
	__m256d zero = _mm256_setzero_pd();
	__m256d* out = MM_MALLOC(__m256d, fftsize * 2);
	if (in.m_digitnum & 3)
	{
		for (i = in.m_digitnum; i <= (in.m_digitnum | 3); i++)
		{
			in.m_digits[i] = 0;
		}
	}
	int digitnum = (in.m_digitnum +3) / 4;
	if (digitnum <= fftsize / 2)
	{
		__m128i* s1 = (__m128i*)in.m_digits;
		__m128i* s2 = (__m128i*)(in.m_digits+fftsize);
		__m256d* d1 = out;
		__m256d* d2 = out + fftsize / 2;
		__m256d* d3 = out + fftsize;
		__m256d* d4 = out + fftsize + fftsize / 2;
		for (i = 0; i < digitnum - fftsize / 4 ; i++)
		{
			__m256d a = _mm256_cvtepi32_pd(s1[i]);
			__m256d b = _mm256_cvtepi32_pd(s2[i]);
			d1[i * 2] = _mm256_add_pd(a, b);
			d1[i * 2+1] = zero;
			d2[i * 2] = _mm256_sub_pd(a, b);
			d2[i * 2 + 1] = zero;
			d3[i * 2] = a;
			d3[i * 2 + 1] = b;
			d4[i * 2] = a;
			d4[i * 2 + 1] = _mm256_sub_pd(zero, b);
		}
		int loopend = std::min(fftsize / 4, digitnum);
		
		for (; i < loopend; i++)
		{
			__m256d a = _mm256_cvtepi32_pd(s1[i]);
			d1[i * 2] = a;
			d1[i * 2 + 1] = zero;
			d2[i * 2] = a;
			d2[i * 2 + 1] = zero;
			d3[i * 2] = a;
			d3[i * 2 + 1] = zero;
			d4[i * 2] = a;
			d4[i * 2 + 1] = zero;
		}
		for (; i < fftsize / 4; i++)
		{
			d1[i * 2] = zero;
			d1[i * 2 + 1] = zero;
			d2[i * 2] = zero;
			d2[i * 2 + 1] = zero;
			d3[i * 2] = zero;
			d3[i * 2 + 1] = zero;
			d4[i * 2] = zero;
			d4[i * 2 + 1] = zero;
		}
	}
	else
	{
		__m128i* s1 = (__m128i*)in.m_digits;
		__m128i* s2 = (__m128i*)(in.m_digits + fftsize * 2);
		__m256d* d1 = out;
		__m256d* d2 = out + fftsize;
		for (i = 0; i < digitnum - fftsize / 2; i++)
		{
			__m256d a = _mm256_cvtepi32_pd(s1[i]);
			__m256d b = _mm256_cvtepi32_pd(s2[i]);
			d1[i * 2] = _mm256_add_pd(a, b);
			d2[i * 2] = _mm256_sub_pd(a, b);
		}
		for (; i < fftsize / 2; i++)
		{
			__m256d a = _mm256_cvtepi32_pd(s1[i]);
			d1[i * 2] = a;
			d2[i * 2] = a;
		}
		d2 = out + fftsize / 2;
		for (i = 0; i < fftsize / 4; i++)
		{
			__m256d a = _mm256_add_pd(d1[i * 2], d2[i * 2]);
			__m256d b = _mm256_sub_pd(d1[i * 2], d2[i * 2]);
			d1[i * 2] = a;
			d1[i * 2 + 1] = zero;
			d2[i * 2] = b;
			d2[i * 2 + 1] = zero;
		}
		d1 = out + fftsize;
		d2 = d1 + fftsize / 2;
		for (i = 0; i < fftsize / 4; i++)
		{
			d1[i * 2 + 1] = d2[i * 2];
			d2[i * 2 + 1] = _mm256_sub_pd(zero, d2[i * 2]);
			//d1[i * 2] = d1[i * 2];
			d2[i * 2] = d1[i * 2];
		}
	}
	int size;
	for (size = fftsize / 8; size > 1; size /= 2)
	{
		for (j = 0; j < fftsize / size / 2; j++)
		{
			__m256d* s1 = out + j * size * 4;
			__m256d* s2 = s1 + size * 2;
			__m256d wr = _mm256_broadcastsd_pd(_mm_load1_pd(WR + j));
			__m256d wi = _mm256_broadcastsd_pd(_mm_load1_pd(WI + j));
			for (i = 0; i < size; i++)
			{
				__m256d ar = s1[i * 2];
				__m256d ai = s1[i * 2 + 1];
				__m256d br0 = s2[i * 2];
				__m256d bi = s2[i * 2 + 1];
				__m256d bbr = _mm256_mul_pd(br0, wr);
				__m256d br = _mm256_fnmadd_pd(bi, wi, bbr);
				bi = _mm256_mul_pd(bi, wr);
				bi = _mm256_fmadd_pd(br0, wi, bi);
				s1[i * 2 + 0] = _mm256_add_pd(ar, br);
				s1[i * 2 + 1] = _mm256_add_pd(ai, bi);
				s2[i * 2 + 0] = _mm256_sub_pd(ar, br);
				s2[i * 2 + 1] = _mm256_sub_pd(ai, bi);
			}

		}
	}
	{
		__m256d* s1 = out;
		for (i = 0; i < fftsize / 2; i++)
		{
			__m256d ar = s1[i * 4];
			__m256d ai = s1[i * 4 + 1];
			__m256d br0 = s1[i * 4 + 2];
			__m256d bi = s1[i * 4 + 3];
			__m256d wr = _mm256_broadcastsd_pd(_mm_load1_pd(WR + i));
			__m256d wi = _mm256_broadcastsd_pd(_mm_load1_pd(WI + i));
			__m256d bbr = _mm256_mul_pd(br0, wr);
			__m256d br = _mm256_fnmadd_pd(bi, wi, bbr);
			bi = _mm256_mul_pd(bi, wr);
			bi = _mm256_fmadd_pd(br0, wi, bi);
			s1[i * 4 + 0] = _mm256_add_pd(ar, br);
			s1[i * 4 + 1] = _mm256_add_pd(ai, bi);
			s1[i * 4 + 2] = _mm256_sub_pd(ar, br);
			s1[i * 4 + 3] = _mm256_sub_pd(ai, bi);
		}
	}
	{
		__m128d* s1 = (__m128d*) out;
		for (i = 0; i < fftsize; i++)
		{
			__m128d ar = s1[i * 4];
			__m128d ai = s1[i * 4 + 2];
			__m128d br0 = s1[i * 4 + 1];
			__m128d bi = s1[i * 4 + 3];
			__m128d wr = _mm_loaddup_pd(WR + i);
			__m128d wi = _mm_loaddup_pd(WI + i);
			__m128d bbr = _mm_mul_pd(br0, wr);
			__m128d br = _mm_fnmadd_pd(bi, wi, bbr);
			bi = _mm_mul_pd(bi, wr);
			bi = _mm_fmadd_pd(br0, wi, bi);
			s1[i * 4 + 0] = _mm_add_pd(ar, br);
			s1[i * 4 + 2] = _mm_add_pd(ai, bi);
			s1[i * 4 + 1] = _mm_sub_pd(ar, br);
			s1[i * 4 + 3] = _mm_sub_pd(ai, bi);
		}
	}
	{
		double* s1 = (double*)out;
		for (i = 0; i < fftsize; i++)
		{
			double ar = s1[i * 8 + 0];
			double ai = s1[i * 8 + 4];
			double br0 = s1[i * 8 + 1];
			double bi = s1[i * 8 + 5];
			double wr = WR[i * 2];
			double wi = WI[i * 2];
			double br = br0 * wr - bi * wi;
			bi = bi * wr  + br0 * wi;
			s1[i * 8 + 0] = ar + br;
			s1[i * 8 + 4] = ai + bi;
			s1[i * 8 + 1] = ar - br;
			s1[i * 8 + 5] = ai - bi;

			ar = s1[i * 8 + 2];
			ai = s1[i * 8 + 6];
			br0 = s1[i * 8 + 3];
			bi = s1[i * 8 + 7];
			wr = WR[i * 2 + 1];
			wi = WI[i * 2 + 1];
			br = br0 * wr - bi * wi;
			bi = bi * wr + br0 * wi;
			s1[i * 8 + 2] = ar + br;
			s1[i * 8 + 6] = ai + bi;
			s1[i * 8 + 3] = ar - br;
			s1[i * 8 + 7] = ai - bi;
		}
	}
	BitFlip(out, fftsize);
	return out;
}

void IFFT(__m256d* in, int fftsize)
{
	int i, j;
	__m256d zero = _mm256_setzero_pd();
	__m256d* out = in;

	int size;
	size = fftsize / 2;
	__m256d* s1 = in;
	__m256d* s2 = s1 + size * 2;
	__m256d* d1 = out;
	__m256d* d2 = d1 + size * 2;
	for (i = 0; i < size; i++)
	{
		__m256d ar = s1[i * 2];
		__m256d ai = s1[i * 2 + 1];
		__m256d br = s2[i * 2];
		__m256d bi = s2[i * 2 + 1];
		d1[i * 2 + 0] = _mm256_add_pd(ar, br);
		d1[i * 2 + 1] = _mm256_add_pd(ai, bi);
		d2[i * 2 + 0] = _mm256_sub_pd(ar, br);
		d2[i * 2 + 1] = _mm256_sub_pd(ai, bi);
	}
	for (size = fftsize / 4; size > 1; size /= 2)
	{
		for (j = 0; j < fftsize / size / 2; j++)
		{
			__m256d* s1 = out + j * size * 4;
			__m256d* s2 = s1 + size * 2;
			for (i = 0; i < size; i++)
			{
				__m256d ar = s1[i * 2];
				__m256d ai = s1[i * 2 + 1];
				__m256d br0 = s2[i * 2];
				__m256d bi = s2[i * 2 + 1];
				__m256d wr = _mm256_broadcastsd_pd(_mm_load1_pd(WR + j));
				__m256d wi = _mm256_broadcastsd_pd(_mm_load1_pd(WI + j));
				__m256d bbr = _mm256_mul_pd(br0, wr);
				__m256d br = _mm256_fmadd_pd(bi, wi, bbr);
				bi = _mm256_mul_pd(bi, wr);
				bi = _mm256_fnmadd_pd(br0, wi, bi);
				s1[i * 2 + 0] = _mm256_add_pd(ar, br);
				s1[i * 2 + 1] = _mm256_add_pd(ai, bi);
				s2[i * 2 + 0] = _mm256_sub_pd(ar, br);
				s2[i * 2 + 1] = _mm256_sub_pd(ai, bi);
			}
		}
	}
	{
		__m256d* s1 = out;
		for (i = 0; i < fftsize / 2; i++)
		{
			__m256d ar = s1[i * 4];
			__m256d ai = s1[i * 4 + 1];
			__m256d br0 = s1[i * 4 + 2];
			__m256d bi = s1[i * 4 + 3];
			__m256d wr = _mm256_broadcastsd_pd(_mm_load1_pd(WR + i));
			__m256d wi = _mm256_broadcastsd_pd(_mm_load1_pd(WI + i));
			__m256d bbr = _mm256_mul_pd(br0, wr);
			__m256d br = _mm256_fmadd_pd(bi, wi, bbr);
			bi = _mm256_mul_pd(bi, wr);
			bi = _mm256_fnmadd_pd(br0, wi, bi);
			s1[i * 4 + 0] = _mm256_add_pd(ar, br);
			s1[i * 4 + 1] = _mm256_add_pd(ai, bi);
			s1[i * 4 + 2] = _mm256_sub_pd(ar, br);
			s1[i * 4 + 3] = _mm256_sub_pd(ai, bi);
		}
	}
	{
		__m128d* s1 = (__m128d*) out;
		for (i = 0; i < fftsize; i++)
		{
			__m128d ar = s1[i * 4];
			__m128d ai = s1[i * 4 + 2];
			__m128d br0 = s1[i * 4 + 1];
			__m128d bi = s1[i * 4 + 3];
			__m128d wr = _mm_loaddup_pd(WR + i);
			__m128d wi = _mm_loaddup_pd(WI + i);
			__m128d bbr = _mm_mul_pd(br0, wr);
			__m128d br = _mm_fmadd_pd(bi, wi, bbr);
			bi = _mm_mul_pd(bi, wr);
			bi = _mm_fnmadd_pd(br0, wi, bi);
			s1[i * 4 + 0] = _mm_add_pd(ar, br);
			s1[i * 4 + 2] = _mm_add_pd(ai, bi);
			s1[i * 4 + 1] = _mm_sub_pd(ar, br);
			s1[i * 4 + 3] = _mm_sub_pd(ai, bi);
		}
	}
	{
		double* s1 = (double*)out;
		for (i = 0; i < fftsize; i++)
		{
			double ar = s1[i * 8 + 0];
			double ai = s1[i * 8 + 4];
			double br0 = s1[i * 8 + 1];
			double bi = s1[i * 8 + 5];
			double wr = WR[i * 2];
			double wi = WI[i * 2];
			double br = br0 * wr + bi * wi;
			bi = bi * wr - br0 * wi;
			s1[i * 8 + 0] = ar + br;
			s1[i * 8 + 4] = ai + bi;
			s1[i * 8 + 1] = ar - br;
			s1[i * 8 + 5] = ai - bi;

			ar = s1[i * 8 + 2];
			ai = s1[i * 8 + 6];
			br0 = s1[i * 8 + 3];
			bi = s1[i * 8 + 7];
			wr = WR[i * 2 + 1];
			wi = WI[i * 2 + 1];
			br = br0 * wr + bi * wi;
			bi = bi * wr - br0 * wi;
			s1[i * 8 + 2] = ar + br;
			s1[i * 8 + 6] = ai + bi;
			s1[i * 8 + 3] = ar - br;
			s1[i * 8 + 7] = ai - bi;
		}
	}
	BitFlip(in, fftsize);
}

__m256d* CompFFT(const CBigInt& re, const CBigInt& im, int fftsize)
{
	int i, j;
	__m256d zero = _mm256_setzero_pd();
	__m256d* out = MM_MALLOC(__m256d, fftsize * 2);
	if (re.m_digitnum & 3)
	{
		for (i = re.m_digitnum; i <= (re.m_digitnum | 3); i++)
		{
			re.m_digits[i] = 0;
		}
	}
	if (im.m_digitnum & 3)
	{
		for (i = im.m_digitnum; i <= (im.m_digitnum | 3); i++)
		{
			im.m_digits[i] = 0;
		}
	}
	{
		int digitnum = (re.m_digitnum + 3) / 4;
		__m128i* s1 = (__m128i*)re.m_digits;
		__m128i* s2 = (__m128i*)(re.m_digits + fftsize * 2);
		__m256d* d1 = out;
		__m256d* d2 = out + fftsize;
		for (i = 0; i < digitnum - fftsize / 2; i++)
		{
			__m256d a = _mm256_cvtepi32_pd(s1[i]);
			__m256d b = _mm256_cvtepi32_pd(s2[i]);
			d1[i * 2] = _mm256_add_pd(a, b);
			d2[i * 2] = _mm256_sub_pd(a, b);
		}
		int loopend = std::min(fftsize / 2, digitnum);
		for (; i < loopend; i++)
		{
			__m256d a = _mm256_cvtepi32_pd(s1[i]);
			d1[i * 2] = a;
			d2[i * 2] = a;
		}
		for (; i < loopend; i++)
		{
			d1[i * 2] = zero;
			d2[i * 2] = zero;
		}
		digitnum = (im.m_digitnum + 3) / 4;
		s1 = (__m128i*)im.m_digits;
		s2 = (__m128i*)(im.m_digits + fftsize * 2);
		d1 = out;
		d2 = out + fftsize;
		if (re.m_sign == im.m_sign)
		{
			for (i = 0; i < digitnum - fftsize / 2; i++)
			{
				__m256d a = _mm256_cvtepi32_pd(s1[i]);
				__m256d b = _mm256_cvtepi32_pd(s2[i]);
				d1[i * 2] = _mm256_add_pd(a, b);
				d2[i * 2] = _mm256_sub_pd(a, b);
			}
			int loopend = std::min(fftsize / 2, digitnum);
			for (; i < loopend; i++)
			{
				__m256d a = _mm256_cvtepi32_pd(s1[i]);
				d1[i * 2] = a;
				d2[i * 2] = a;
			}
		}
		else
		{
			for (i = 0; i < digitnum - fftsize / 2; i++)
			{
				__m256d a = _mm256_cvtepi32_pd(s1[i]);
				__m256d b = _mm256_cvtepi32_pd(s2[i]);
				d1[i * 2] = _mm256_sub_pd(a, b);
				d2[i * 2] = _mm256_add_pd(a, b);
			}
			int loopend = std::min(fftsize / 2, digitnum);
			for (; i < loopend; i++)
			{
				__m256d a = _mm256_sub_pd(zero, _mm256_cvtepi32_pd(s1[i]));
				d1[i * 2] = a;
				d2[i * 2] = a;
			}
		}
		for (; i < loopend; i++)
		{
			d1[i * 2] = zero;
			d2[i * 2] = zero;
		}
	}
	int size;
	for (size = fftsize / 4; size > 1; size /= 2)
	{
		for (j = 0; j < fftsize / size / 2; j++)
		{
			__m256d* s1 = out + j * size * 4;
			__m256d* s2 = s1 + size * 2;
			__m256d wr = _mm256_broadcastsd_pd(_mm_load1_pd(WR + j));
			__m256d wi = _mm256_broadcastsd_pd(_mm_load1_pd(WI + j));
			for (i = 0; i < size; i++)
			{
				__m256d ar = s1[i * 2];
				__m256d ai = s1[i * 2 + 1];
				__m256d br0 = s2[i * 2];
				__m256d bi = s2[i * 2 + 1];
				__m256d bbr = _mm256_mul_pd(br0, wr);
				__m256d br = _mm256_fnmadd_pd(bi, wi, bbr);
				bi = _mm256_mul_pd(bi, wr);
				bi = _mm256_fmadd_pd(br0, wi, bi);
				s1[i * 2 + 0] = _mm256_add_pd(ar, br);
				s1[i * 2 + 1] = _mm256_add_pd(ai, bi);
				s2[i * 2 + 0] = _mm256_sub_pd(ar, br);
				s2[i * 2 + 1] = _mm256_sub_pd(ai, bi);
			}
		}
	}
	{
		__m256d* s1 = out;
		for (i = 0; i < fftsize / 2; i++)
		{
			__m256d ar = s1[i * 4];
			__m256d ai = s1[i * 4 + 1];
			__m256d br0 = s1[i * 4 + 2];
			__m256d bi = s1[i * 4 + 3];
			__m256d wr = _mm256_broadcastsd_pd(_mm_load1_pd(WR + i));
			__m256d wi = _mm256_broadcastsd_pd(_mm_load1_pd(WI + i));
			__m256d bbr = _mm256_mul_pd(br0, wr);
			__m256d br = _mm256_fnmadd_pd(bi, wi, bbr);
			bi = _mm256_mul_pd(bi, wr);
			bi = _mm256_fmadd_pd(br0, wi, bi);
			s1[i * 4 + 0] = _mm256_add_pd(ar, br);
			s1[i * 4 + 1] = _mm256_add_pd(ai, bi);
			s1[i * 4 + 2] = _mm256_sub_pd(ar, br);
			s1[i * 4 + 3] = _mm256_sub_pd(ai, bi);
		}
	}
	{
		__m128d* s1 = (__m128d*) out;
		for (i = 0; i < fftsize; i++)
		{
			__m128d ar = s1[i * 4];
			__m128d ai = s1[i * 4 + 2];
			__m128d br0 = s1[i * 4 + 1];
			__m128d bi = s1[i * 4 + 3];
			__m128d wr = _mm_loaddup_pd(WR + i);
			__m128d wi = _mm_loaddup_pd(WI + i);
			__m128d bbr = _mm_mul_pd(br0, wr);
			__m128d br = _mm_fnmadd_pd(bi, wi, bbr);
			bi = _mm_mul_pd(bi, wr);
			bi = _mm_fmadd_pd(br0, wi, bi);
			s1[i * 4 + 0] = _mm_add_pd(ar, br);
			s1[i * 4 + 2] = _mm_add_pd(ai, bi);
			s1[i * 4 + 1] = _mm_sub_pd(ar, br);
			s1[i * 4 + 3] = _mm_sub_pd(ai, bi);
		}
	}
	{
		double* s1 = (double*)out;
		for (i = 0; i < fftsize; i++)
		{
			double ar = s1[i * 8 + 0];
			double ai = s1[i * 8 + 4];
			double br0 = s1[i * 8 + 1];
			double bi = s1[i * 8 + 5];
			double wr = WR[i * 2];
			double wi = WI[i * 2];
			double br = br0 * wr - bi * wi;
			bi = bi * wr + br0 * wi;
			s1[i * 8 + 0] = ar + br;
			s1[i * 8 + 4] = ai + bi;
			s1[i * 8 + 1] = ar - br;
			s1[i * 8 + 5] = ai - bi;

			ar = s1[i * 8 + 2];
			ai = s1[i * 8 + 6];
			br0 = s1[i * 8 + 3];
			bi = s1[i * 8 + 7];
			wr = WR[i * 2 + 1];
			wi = WI[i * 2 + 1];
			br = br0 * wr - bi * wi;
			bi = bi * wr + br0 * wi;
			s1[i * 8 + 2] = ar + br;
			s1[i * 8 + 6] = ai + bi;
			s1[i * 8 + 3] = ar - br;
			s1[i * 8 + 7] = ai - bi;
		}
	}
	BitFlip(out, fftsize);
	return out;
}

void BitFlip(__m256d* in, int size)
{
	double* buf = (double*)in;
	size *= 4;
	auto calcaddr = [](int addr) {
		return (addr & 3) + (addr & (~3)) * 2;
	};
	if (size > 65536)
	{
		int div = int(0x100000000ull / size);
		for (int i = 0; i < size; i++)
		{
			int flip = (unsigned int)(bitflip[i & 65535] * 65536 + bitflip[i >>16]) / div;
			if (i < flip)
			{
				int a = calcaddr(i);
				int b = calcaddr(flip);
				double t = buf[a];
				buf[a] = buf[b];
				buf[b] = t;
				t = buf[a + 4];
				buf[a + 4] = buf[b + 4];
				buf[b + 4] = t;
			}
		}
	}
	else
	{
		int div = 65536 / size;
		for (int i = 0; i < size; i++)
		{
			int flip = bitflip[i] / div;
			if (i < flip)
			{
				int a = calcaddr(i);
				int b = calcaddr(flip);
				double t = buf[a];
				buf[a] = buf[b];
				buf[b] = t;
				t = buf[a + 4];
				buf[a + 4] = buf[b + 4];
				buf[b + 4] = t;
			}
		}
	}
}
