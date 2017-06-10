#include "BigInt.h"

#include <algorithm>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include "BigInt.h"

namespace
{
	void Karatsuba4(const int* bufa, const int* bufb, int* buf);
	void Karatsuba8(const int* bufa, const int* bufb, int* buf);
	void Karatsuba16(const int* bufa, const int* bufb, int* buf);
	void Karatsuba32(const int* bufa, const int* bufb, int* buf);

	void Karatsuba4n(const int* bufa, const int* bufb, int sizeb, int* buf);
	void Karatsuba8n(const int* bufa, const int* bufb, int sizeb, int* buf);




	void Karatsuba4(const int* bufa, const int* bufb, int* buf)
	{
		__m128i a = _mm_load_si128((__m128i*)bufa);
		__m128i b = _mm_load_si128((__m128i*)bufb);
		__m128i odd = _mm_mullo_epi32(a, b);
		__m128i odd2 = _mm_mullo_epi32(_mm_alignr_epi8(a, a, 8), b);
		odd2 = _mm_add_epi32(_mm_alignr_epi8(odd2, odd2, 8), odd2);
		odd2 = _mm_srli_si128(_mm_slli_si128(odd2, 8), 4);
		odd = _mm_add_epi32(odd, odd2);
		__m128i even = _mm_mullo_epi32(_mm_alignr_epi8(a, a, 12), b);
		__m128i even2 = _mm_mullo_epi32(_mm_alignr_epi8(b, b, 12), a);
		even = _mm_add_epi32(even, even2);
		even = _mm_add_epi32(_mm_srli_si128(_mm_slli_si128(even, 12), 8), _mm_srli_si128(even, 4));
		__m128i hi = _mm_unpackhi_epi32(odd, even);
		__m128i lo = _mm_unpacklo_epi32(odd, even);
		_mm_store_si128((__m128i*)buf, lo);
		_mm_store_si128(((__m128i*)buf) + 1, hi);
	}

	void Karatsuba8(const int* bufa, const int* bufb, int* buf)
	{
		__m256i ahi = _mm256_castps_si256(_mm256_broadcast_ps(((__m128*)bufa) + 1));
		__m256i alo = _mm256_castps_si256(_mm256_broadcast_ps((__m128*)bufa));
		__m256i b = _mm256_load_si256((__m256i*)bufb);
		__m256i odd = _mm256_mullo_epi32(alo, b);
		__m256i odd2 = _mm256_mullo_epi32(_mm256_alignr_epi8(alo, alo, 8), b);
		odd2 = _mm256_add_epi32(_mm256_alignr_epi8(odd2, odd2, 8), odd2);
		odd2 = _mm256_srli_si256(_mm256_slli_si256(odd2, 8), 4);
		odd = _mm256_add_epi32(odd, odd2);
		__m256i even = _mm256_mullo_epi32(_mm256_alignr_epi8(alo, alo, 12), b);
		__m256i even2 = _mm256_mullo_epi32(_mm256_alignr_epi8(b, b, 12), alo);
		even = _mm256_add_epi32(even, even2);
		even = _mm256_add_epi32(_mm256_srli_si256(_mm256_slli_si256(even, 12), 8), _mm256_srli_si256(even, 4));
		__m256i md = _mm256_unpackhi_epi32(odd, even);
		__m256i lo = _mm256_unpacklo_epi32(odd, even);

		odd = _mm256_mullo_epi32(ahi, b);
		odd2 = _mm256_mullo_epi32(_mm256_alignr_epi8(ahi, ahi, 8), b);
		odd2 = _mm256_add_epi32(_mm256_alignr_epi8(odd2, odd2, 8), odd2);
		odd2 = _mm256_srli_si256(_mm256_slli_si256(odd2, 8), 4);
		odd = _mm256_add_epi32(odd, odd2);
		even = _mm256_mullo_epi32(_mm256_alignr_epi8(ahi, ahi, 12), b);
		even2 = _mm256_mullo_epi32(_mm256_alignr_epi8(b, b, 12), ahi);
		even = _mm256_add_epi32(even, even2);
		even = _mm256_add_epi32(_mm256_srli_si256(_mm256_slli_si256(even, 12), 8), _mm256_srli_si256(even, 4));
		__m256i hi = _mm256_unpacklo_epi32(odd, even);
		md = _mm256_add_epi32(md, hi);
		hi = _mm256_unpackhi_epi32(odd, even);
		lo = _mm256_add_epi32(lo, _mm256_permute2x128_si256(md, md, 0x08));
		hi = _mm256_add_epi32(hi, _mm256_permute2x128_si256(md, md, 0x81));
		_mm256_store_si256((__m256i*)buf, lo);
		_mm256_store_si256(((__m256i*)buf) + 1, hi);
	}

	void Karatsuba16(const int* bufa, const int* bufb, int* buf)
	{
		Karatsuba8(bufa, bufb, buf);
		Karatsuba8(bufa + 8, bufb + 8, buf + 16);
		__m256i a = _mm256_load_si256((__m256i*)bufa);
		__m256i b = _mm256_load_si256((__m256i*)(bufa + 8));
		__m256i c = _mm256_load_si256((__m256i*)bufb);
		__m256i d = _mm256_load_si256((__m256i*)(bufb + 8));
		a = _mm256_sub_epi32(a, b);
		c = _mm256_sub_epi32(c, d);
		_mm256_store_si256((__m256i*)(buf + 32), a);
		_mm256_store_si256((__m256i*)(buf + 40), c);
		Karatsuba8(buf+32, buf+40, buf+48);
		a = _mm256_load_si256((__m256i*)(buf + 0));
		b = _mm256_load_si256((__m256i*)(buf + 8));
		c = _mm256_load_si256((__m256i*)(buf + 16));
		d = _mm256_load_si256((__m256i*)(buf + 24));
		__m256i x1 = _mm256_load_si256((__m256i*)(buf + 48));
		__m256i x2 = _mm256_load_si256((__m256i*)(buf + 56));
		x1 = _mm256_sub_epi32(x1, a);
		x1 = _mm256_sub_epi32(x1, c);
		x2 = _mm256_sub_epi32(x2, b);
		x2 = _mm256_sub_epi32(x2, d);
		b = _mm256_sub_epi32(b, x1);
		c = _mm256_sub_epi32(c, x2);
		_mm256_store_si256((__m256i*)(buf + 8), b);
		_mm256_store_si256((__m256i*)(buf + 16), c);
	}

	void Karatsuba32(const int* bufa, const int* bufb, int* buf)
	{
		Karatsuba16(bufa, bufb, buf);
		Karatsuba16(bufa + 16, bufb + 16, buf + 32);
		__m256i a, b, c, d, x1, x2;
		a = _mm256_load_si256((__m256i*)bufa);
		b = _mm256_load_si256((__m256i*)(bufa + 16));
		c = _mm256_load_si256((__m256i*)bufb);
		d = _mm256_load_si256((__m256i*)(bufb + 16));
		a = _mm256_sub_epi32(a, b);
		c = _mm256_sub_epi32(c, d);
		_mm256_store_si256((__m256i*)(buf + 64), a);
		_mm256_store_si256((__m256i*)(buf + 80), c);
		a = _mm256_load_si256((__m256i*)(bufa + 8));
		b = _mm256_load_si256((__m256i*)(bufa + 24));
		c = _mm256_load_si256((__m256i*)(bufb + 8));
		d = _mm256_load_si256((__m256i*)(bufb + 24));
		a = _mm256_sub_epi32(a, b);
		c = _mm256_sub_epi32(c, d);
		_mm256_store_si256((__m256i*)(buf + 72), a);
		_mm256_store_si256((__m256i*)(buf + 88), c);
		Karatsuba16(buf + 64, buf + 80, buf + 96);
		a = _mm256_load_si256((__m256i*)(buf + 0));
		b = _mm256_load_si256((__m256i*)(buf + 16));
		c = _mm256_load_si256((__m256i*)(buf + 32));
		d = _mm256_load_si256((__m256i*)(buf + 48));
		x1 = _mm256_load_si256((__m256i*)(buf + 96));
		x2 = _mm256_load_si256((__m256i*)(buf + 112));
		x1 = _mm256_sub_epi32(x1, a);
		x1 = _mm256_sub_epi32(x1, c);
		x2 = _mm256_sub_epi32(x2, b);
		x2 = _mm256_sub_epi32(x2, d);
		b = _mm256_sub_epi32(b, x1);
		c = _mm256_sub_epi32(c, x2);
		_mm256_store_si256((__m256i*)(buf + 16), b);
		_mm256_store_si256((__m256i*)(buf + 32), c);
		a = _mm256_load_si256((__m256i*)(buf + 8));
		b = _mm256_load_si256((__m256i*)(buf + 24));
		c = _mm256_load_si256((__m256i*)(buf + 40));
		d = _mm256_load_si256((__m256i*)(buf + 56));
		x1 = _mm256_load_si256((__m256i*)(buf + 104));
		x2 = _mm256_load_si256((__m256i*)(buf + 120));
		x1 = _mm256_sub_epi32(x1, a);
		x1 = _mm256_sub_epi32(x1, c);
		x2 = _mm256_sub_epi32(x2, b);
		x2 = _mm256_sub_epi32(x2, d);
		b = _mm256_sub_epi32(b, x1);
		c = _mm256_sub_epi32(c, x2);
		_mm256_store_si256((__m256i*)(buf + 24), b);
		_mm256_store_si256((__m256i*)(buf + 40), c);
	}

	void Karatsuba4n(const int* bufa, const int* bufb, int sizeb, int* buf)
	{
		__m256i carry = _mm256_setzero_si256();
		__m256i a = _mm256_castps_si256(_mm256_broadcast_ps((__m128*)bufa)); 
		int i;
		for (i = 0; i < sizeb; i += 8)
		{
			__m256i b = _mm256_load_si256((__m256i*)(bufb + i));
			__m256i odd = _mm256_mullo_epi32(a, b);
			__m256i odd2 = _mm256_mullo_epi32(_mm256_alignr_epi8(a, a, 8), b);
			odd2 = _mm256_add_epi32(_mm256_alignr_epi8(odd2, odd2, 8), odd2);
			odd2 = _mm256_srli_si256(_mm256_slli_si256(odd2, 8), 4);
			odd = _mm256_add_epi32(odd, odd2);
			__m256i even = _mm256_mullo_epi32(_mm256_alignr_epi8(a, a, 12), b);
			__m256i even2 = _mm256_mullo_epi32(_mm256_alignr_epi8(b, b, 12), a);
			even = _mm256_add_epi32(even, even2);
			even = _mm256_add_epi32(_mm256_srli_si256(_mm256_slli_si256(even, 12), 8), _mm256_srli_si256(even, 4));
			__m256i hi = _mm256_unpackhi_epi32(odd, even);
			__m256i lo = _mm256_unpacklo_epi32(odd, even);
			carry = _mm256_inserti128_si256(carry, _mm256_castsi256_si128(hi), 1);
			carry = _mm256_add_epi32(carry, lo);
			_mm256_store_si256((__m256i*)(buf + i), carry);
			carry = _mm256_permute2x128_si256(hi, hi, 0x81);
		}
		_mm256_store_si256((__m256i*)(buf + i), carry);
	}

	void Karatsuba8n(const int* bufa, const int* bufb, int sizeb, int* buf)
	{
		__m256i carry = _mm256_setzero_si256();
		__m256i ahi = _mm256_castps_si256(_mm256_broadcast_ps(((__m128*)bufa) + 1));
		__m256i alo = _mm256_castps_si256(_mm256_broadcast_ps((__m128*)bufa));
		int i;
		for (i = 0; i < sizeb; i += 8)
		{
			__m256i b = _mm256_load_si256((__m256i*)(bufb + i));
			__m256i odd = _mm256_mullo_epi32(alo, b);
			__m256i odd2 = _mm256_mullo_epi32(_mm256_alignr_epi8(alo, alo, 8), b);
			odd2 = _mm256_add_epi32(_mm256_alignr_epi8(odd2, odd2, 8), odd2);
			odd2 = _mm256_srli_si256(_mm256_slli_si256(odd2, 8), 4);
			odd = _mm256_add_epi32(odd, odd2);
			__m256i even = _mm256_mullo_epi32(_mm256_alignr_epi8(alo, alo, 12), b);
			__m256i even2 = _mm256_mullo_epi32(_mm256_alignr_epi8(b, b, 12), alo);
			even = _mm256_add_epi32(even, even2);
			even = _mm256_add_epi32(_mm256_srli_si256(_mm256_slli_si256(even, 12), 8), _mm256_srli_si256(even, 4));
			__m256i md = _mm256_unpackhi_epi32(odd, even);
			__m256i lo = _mm256_unpacklo_epi32(odd, even);
			lo = _mm256_add_epi32(lo, carry);

			odd = _mm256_mullo_epi32(ahi, b);
			odd2 = _mm256_mullo_epi32(_mm256_alignr_epi8(ahi, ahi, 8), b);
			odd2 = _mm256_add_epi32(_mm256_alignr_epi8(odd2, odd2, 8), odd2);
			odd2 = _mm256_srli_si256(_mm256_slli_si256(odd2, 8), 4);
			odd = _mm256_add_epi32(odd, odd2);
			even = _mm256_mullo_epi32(_mm256_alignr_epi8(ahi, ahi, 12), b);
			even2 = _mm256_mullo_epi32(_mm256_alignr_epi8(b, b, 12), ahi);
			even = _mm256_add_epi32(even, even2);
			even = _mm256_add_epi32(_mm256_srli_si256(_mm256_slli_si256(even, 12), 8), _mm256_srli_si256(even, 4));
			carry = _mm256_unpacklo_epi32(odd, even);
			md = _mm256_add_epi32(md, carry);
			carry = _mm256_unpackhi_epi32(odd, even);
			lo = _mm256_add_epi32(lo, _mm256_permute2x128_si256(md, md, 0x08));
			carry = _mm256_add_epi32(carry, _mm256_permute2x128_si256(md, md, 0x81));
			_mm256_store_si256((__m256i*)(buf + i), lo);
		}
		_mm256_store_si256((__m256i*)(buf + i), carry);
	}

}

CBigInt KaratsubaMul(const CBigInt& a, const CBigInt& b)
{
	const CBigInt* s;
	const CBigInt* l;
	if (a.m_digitnum > b.m_digitnum)
	{
		s = &b;
		l = &a;
	}
	else
	{
		s = &a;
		l = &b;
	}
	int sizes;
	int sizel;
	unsigned long sizesb;
#ifdef _MSC_VER
	_BitScanReverse(&sizesb, s->m_digitnum - 1);
#else
	sizesb = 31 - __builtin_clz(s->m_digitnum - 1);
#endif
	/*	if (s->m_digitnum < (3 << (sizesb - 1)))
	{
	sizes = 3 << (sizesb - 1);
	}
	else*/
	{
		sizes = 1 << (sizesb + 1);
		if (sizes < 4)
			sizes = 4;
	}
	int i;
	for (i = s->m_digitnum; i < sizes; i++)
	{
		s->m_digits[i] = 0;
	}
	int sizess = sizes;
	if (sizes == 4)
	{
		sizess = 8;
	}
	sizel = (l->m_digitnum - 1 | sizess - 1) + 1;
	for (i = l->m_digitnum; i < sizel; i++)
	{
		l->m_digits[i] = 0;
	}
	int memsize = NextMemSize(sizel * 3 + sizes * 8 + 7);
	int* buf;
	buf = MM_MALLOC(int, memsize);
	switch (sizes)
	{
	case 4:
		if (sizel == 4)
		{
			Karatsuba4(s->m_digits, l->m_digits, buf);
		}
		else
		{
			Karatsuba4n(s->m_digits, l->m_digits, sizel, buf);
		}
		break;
	case 8:
		if (sizel == 8)
		{
			Karatsuba8(s->m_digits, l->m_digits, buf);
		}
		else
		{
			Karatsuba8n(s->m_digits, l->m_digits, sizel, buf);
		}
		break;
	case 16:
		for (i = 0; i < sizel; i += 16 * 2)
		{
			Karatsuba16(s->m_digits, l->m_digits + i, buf + i);
		}
		if (sizel != 16)
		{
			for (; i < sizel + 16; i++)
			{
				buf[i] = 0;
			}
			int* buf2 = buf + sizel + 16;
			for (i = 16; i < sizel; i += 16 * 2)
			{
				Karatsuba16(s->m_digits, l->m_digits + i, buf2);
				__m256i s1 = _mm256_load_si256((__m256i*)buf2);
				__m256i d1 = _mm256_load_si256((__m256i*)(buf + i));
				__m256i s2 = _mm256_load_si256((__m256i*)(buf2 + 8));
				__m256i d2 = _mm256_load_si256((__m256i*)(buf + i + 8));
				__m256i s3 = _mm256_load_si256((__m256i*)(buf2 + 16));
				__m256i d3 = _mm256_load_si256((__m256i*)(buf + i + 16));
				__m256i s4 = _mm256_load_si256((__m256i*)(buf2 + 24));
				__m256i d4 = _mm256_load_si256((__m256i*)(buf + i + 24));
				d1 = _mm256_add_epi32(d1, s1);
				d2 = _mm256_add_epi32(d2, s2);
				d3 = _mm256_add_epi32(d3, s3);
				d4 = _mm256_add_epi32(d4, s4);
				_mm256_store_si256((__m256i*)(buf + i), d1);
				_mm256_store_si256((__m256i*)(buf + i + 8), d2);
				_mm256_store_si256((__m256i*)(buf + i + 16), d3);
				_mm256_store_si256((__m256i*)(buf + i + 24), d4);
			}
		}
		break;
	case 32:
		for (i = 0; i < sizel; i += 32 * 2)
		{
			Karatsuba32(s->m_digits, l->m_digits + i, buf + i);
		}
		if (sizel != 32)
		{
			for (; i < sizel + 32; i++)
			{
				buf[i] = 0;
			}
			int* buf2 = buf + sizel + 32;
			for (i = 32; i < sizel; i += 32 * 2)
			{
				Karatsuba32(s->m_digits, l->m_digits + i, buf2);
				int j;
				for (j = 0; j < 64; j += 8)
				{
					__m256i s1 = _mm256_load_si256((__m256i*)(buf2 + j));
					__m256i d1 = _mm256_load_si256((__m256i*)(buf + i + j));
					d1 = _mm256_add_epi32(d1, s1);
					_mm256_store_si256((__m256i*)(buf + i + j), d1);
				}
			}
		}
		break;
	}
	CBigInt ret;
	ret.m_maxdigitnum = memsize;
	ret.m_digits = buf;
	ret.m_sign = a.m_sign * b.m_sign;
	unsigned int carry = 0;
	unsigned int* uiBuf = (unsigned int*)buf;
	for (i = 0; i < a.m_digitnum + b.m_digitnum; i++)
	{
		unsigned int data = carry + uiBuf[i];
		uiBuf[i] = data % BIGINT_BASE;
		if (uiBuf[i])
		{
			ret.m_digitnum = i + 1;
		}
		carry = data / BIGINT_BASE;
	}
	return ret;
}
