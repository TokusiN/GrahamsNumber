#include <stdio.h>
#include <algorithm>
#include <vector>
#include <memory.h>
#include <time.h>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "BigInt.h"

void FFTInit(int size);
CBigInt FFTMul(const CBigInt& a, const CBigInt& b);
CBigInt KaratsubaMul(const CBigInt& a, const CBigInt& b);

struct TMallocList
{
	void* ptr;
	int size;
};

void BigIntInit(int size)
{
	FFTInit(size);
}

void* AlignedMalloc(size_t size)
{
	void* ptr = _mm_malloc(size, 64);
	return ptr;
}

void AlignedFree(void* ptr)
{
	_mm_free(ptr);
}


int NextMemSize(int n)
{
	n |= 31;
	unsigned long s;
#ifdef _MSC_VER
	_BitScanReverse(&s, n);
#else
	s = 31 - __builtin_clz(n);
#endif
	return 1 << (s + 1);
}

int NextPow2(int n)
{
	unsigned long s;
#ifdef _MSC_VER
	_BitScanReverse(&s, n);
#else
	s = 31 - __builtin_clz(n);
#endif
	return 1 << (s + 1);
}

CBigInt::CBigInt()
{
	m_digitnum = 0;
	m_digits = NULL;
	m_maxdigitnum = 0;
	m_sign = 1;
}

CBigInt::CBigInt(const CBigInt& in)
{
	m_maxdigitnum = in.m_maxdigitnum;
	m_digits = MM_MALLOC(int, m_maxdigitnum);
	m_sign = in.m_sign;
	m_digitnum = in.m_digitnum;
	memcpy(m_digits, in.m_digits, sizeof(int) * in.m_digitnum);
}

CBigInt::CBigInt(CBigInt&& in)
{
	m_maxdigitnum = in.m_maxdigitnum;
	m_digits = in.m_digits;
	m_digitnum = in.m_digitnum;
	m_sign = in.m_sign;
	in.m_digits = NULL;
}

template <class T> CBigInt::CBigInt(T in)
{
	m_maxdigitnum = 32;
	m_digits = MM_MALLOC(int, m_maxdigitnum);
	m_sign = 1;
	if (in < 0)
	{
		m_sign = -1;
		in = -in;
		if (in < 0)
		{
			abort();
		}
	}
	int i;
	for (i = 0; in; i++)
	{
		m_digits[i] = in % BIGINT_BASE;
		in /= BIGINT_BASE;
	}
	m_digitnum = i;
}


template CBigInt::CBigInt (int in);
template CBigInt::CBigInt (long in);
template CBigInt::CBigInt (long long in);
template CBigInt::CBigInt(unsigned int in);
template CBigInt::CBigInt(unsigned long in);
template CBigInt::CBigInt(unsigned long long in);

CBigInt::CBigInt(const char* in)
{
	int i, j = 0;
	bool inputed = false;
	for (i = 0; in[i]; i++)
	{
		if (in[i] >= '0' && in[i] <= '9')
		{
			inputed = true;
		}
		else
		{
			if (inputed)
			{
				break;
			}
		}
	}
	int len = i;
	int digit = NextMemSize(m_maxdigitnum = (len + BIGINT_BASE_DIGITS - 1) / BIGINT_BASE_DIGITS);
	if (m_maxdigitnum < digit)
	{
		if (m_digits) MM_FREE(m_digits);
		m_maxdigitnum = digit;
		m_digits = MM_MALLOC(int, digit);
	}
	m_sign = 1;
	int b = 1;
	m_digitnum = 0;
	for (i--; i >= 0; i--)
	{
		if (in[i] >= '0' && in[i] <= '9')
		{
			int d = in[i] - '0';
			int pos = j / BIGINT_BASE_DIGITS;
			if (j % 4 == 0)
			{
				m_digits[pos] = d;
				b = 1;
			}
			else
			{
				b *= 10;
				m_digits[pos] += b * d;
			}
			if (m_digits[pos] > 0)
			{
				m_digitnum = pos + 1;
			}
			j++;
		}
		else if (in[i] == '-')
		{
			m_sign = -1;
		}
	}
	if (m_digitnum == 0)
	{
		m_sign = 1;
	}
}

CBigInt::~CBigInt()
{
	if (m_digits) MM_FREE(m_digits);
}

CBigInt& CBigInt::operator=(const char* in)
{
	int i, j = 0;
	bool inputed = false;
	for (i = 0; in[i]; i++)
	{
		if (in[i] >= '0' && in[i] <= '9')
		{
			inputed = true;
		}
		else
		{
			if (inputed)
			{
				break;
			}
		}
	}
	int len = i;
	int digit = NextMemSize(m_maxdigitnum = (len + BIGINT_BASE_DIGITS - 1) / BIGINT_BASE_DIGITS);
	if (m_maxdigitnum < digit)
	{
		if (m_digits) MM_FREE(m_digits);
		m_maxdigitnum = digit;
		m_digits = MM_MALLOC(int, digit);
	}
	m_sign = 1;
	int b = 1;
	m_digitnum = 0;
	for (i--; i >= 0; i--)
	{
		if (in[i] >= '0' && in[i] <= '9')
		{
			int d = in[i] - '0';
			int pos = j / BIGINT_BASE_DIGITS;
			if (j % 4 == 0)
			{
				m_digits[pos] = d;
				b = 1;
			}
			else
			{
				b *= 10;
				m_digits[pos] += b * d;
			}
			if (m_digits[pos] > 0)
			{
				m_digitnum = pos + 1;
			}
			j++;
		}
		else if (in[i] == '-')
		{
			m_sign = -1;
		}
	}
	if (m_digitnum == 0)
	{
		m_sign = 1;
	}
	return *this;
}

template <class T> CBigInt& CBigInt::operator= (T in)
{
	int digit = 64;
	if (m_maxdigitnum < digit)
	{
		if (m_digits) MM_FREE(m_digits);
		m_maxdigitnum = digit;
		m_digits = MM_MALLOC(int, digit);
	}
	m_sign = 1;
	if (in < 0)
	{
		m_sign = -1;
		in = -in;
		if (in < 0)
		{
			abort();
		}
	}
	int i;
	for (i = 0; in; i++)
	{
		m_digits[i] = in % BIGINT_BASE;
		in /= BIGINT_BASE;
	}
	m_digitnum = i;
	return *this;
}
template CBigInt& CBigInt::operator= <int>(int in);
template CBigInt& CBigInt::operator= <long>(long in);
template CBigInt& CBigInt::operator= <long long>(long long in);

#ifdef _MSC_VER
template <> CBigInt& CBigInt::operator= (double in)
#else
CBigInt& CBigInt::operator= (double in)
#endif
{
	m_sign = 1;
	if (in < 0)
	{
		m_sign = -1;
		in = -in;
	}
	int digit;
	for (digit = 0; in > 1; digit++)
	{
		in /= BIGINT_BASE;
	}
	m_digitnum = digit;
	if (m_maxdigitnum < digit)
	{
		if (m_digits) MM_FREE(m_digits);
		m_maxdigitnum = NextMemSize(digit);
		m_digits = MM_MALLOC(int, m_maxdigitnum);
	}
	int i;
	for (i = digit - 1; i >= 0; i--)
	{
		in *= BIGINT_BASE;
		int d = floor(in);
		m_digits[i] = d;
		in -= d;
	}
	return *this;
}


CBigInt& CBigInt::operator= (const CBigInt& in)
{
	int digit = NextMemSize(in.m_digitnum);
	if (m_maxdigitnum < digit)
	{
		if (m_digits) MM_FREE(m_digits);
		m_maxdigitnum = digit;
		m_digits = MM_MALLOC(int, digit);
	}
	m_sign = in.m_sign;
	m_digitnum = in.m_digitnum;
	memcpy(m_digits, in.m_digits, sizeof(int) * in.m_digitnum);
	return *this;
}

CBigInt& CBigInt::operator= (CBigInt&& in)
{
	if (m_digits) MM_FREE(m_digits);
	m_maxdigitnum = in.m_maxdigitnum;
	m_digits = in.m_digits;
	m_digitnum = in.m_digitnum;
	m_sign = in.m_sign;
	in.m_digits = NULL;
	return *this;
}

CBigInt& CBigInt::operator+=(const CBigInt& in)
{
	if (m_sign != in.m_sign)
	{
		m_sign = -m_sign;
		*this -= in;
		if (m_digitnum) {
			m_sign = -m_sign;
		}
		return *this;
	}
	if (in.m_digitnum >= m_maxdigitnum - 1 || m_digitnum >= m_maxdigitnum - 1)
	{
		// 桁数拡張あり
		int newdigitnum = NextMemSize(std::max(in.m_digitnum + 2, m_maxdigitnum + 2));
		int* p = MM_MALLOC(int, newdigitnum);
		int i;
		int mindigitnum = std::min(in.m_digitnum, m_digitnum);
		int c = 0;
		for (i = 0; i < mindigitnum; i++)
		{
			p[i] = m_digits[i] + in.m_digits[i] + c;
			c = p[i] / BIGINT_BASE;
			p[i] %= BIGINT_BASE;
		}
		if (in.m_digitnum > mindigitnum)
		{
			for (; i < in.m_digitnum && c; i++)
			{
				p[i] = in.m_digits[i] + c;
				c = p[i] / BIGINT_BASE;
				p[i] %= BIGINT_BASE;
			}
			if (i < in.m_digitnum)
			{
				memcpy(p + i, in.m_digits + i, sizeof(int) * (in.m_digitnum - i));
				m_digitnum = in.m_digitnum;
			}
			else if (c)
			{
				p[i] = 1;
				m_digitnum = i + 1;
			}
			else
			{
				m_digitnum = i;
			}
		}
		else
		{
			for (; i < m_digitnum && c; i++)
			{
				p[i] = m_digits[i] + c;
				c = p[i] / BIGINT_BASE;
				p[i] %= BIGINT_BASE;
			}
			if (i < m_digitnum)
			{
				memcpy(p + i, m_digits + i, sizeof(int) * (m_digitnum - i));
			}
			else if (c)
			{
				p[i] = 1;
				m_digitnum = i + 1;
			}
			else
			{
				m_digitnum = i;
			}
		}
		if (m_digits) MM_FREE(m_digits);
		m_digits = p;
		m_maxdigitnum = newdigitnum;
	}
	else
	{
		// 桁数拡張無し
		int i;
		int mindigitnum = std::min(in.m_digitnum, m_digitnum);
		int c = 0;
		for (i = 0; i < mindigitnum; i++)
		{
			m_digits[i] += in.m_digits[i] + c;
			c = m_digits[i] / BIGINT_BASE;
			m_digits[i] %= BIGINT_BASE;
		}
		if (in.m_digitnum > mindigitnum)
		{
			for (; i<in.m_digitnum && c; i++)
			{
				m_digits[i] = in.m_digits[i] + c;
				c = m_digits[i] / BIGINT_BASE;
				m_digits[i] %= BIGINT_BASE;
			}
			if (i < in.m_digitnum)
			{
				memcpy(m_digits + i, in.m_digits + i, sizeof(int) * (in.m_digitnum - i));
				m_digitnum = in.m_digitnum;
			}
			else if (c)
			{
				m_digits[i] = 1;
				m_digitnum = i + 1;
			}
			else
			{
				m_digitnum = i;
			}
		}
		else
		{
			for (; i < m_digitnum && c; i++)
			{
				m_digits[i] += c;
				c = m_digits[i] / BIGINT_BASE;
				m_digits[i] %= BIGINT_BASE;
			}
			if (c)
			{
				m_digits[i] = 1;
				m_digitnum = i + 1;
			}
		}
	}
	return *this;
}

CBigInt& CBigInt::operator-=(const CBigInt& in)
{
	if (m_sign != in.m_sign)
	{
		m_sign = -m_sign;
		*this += in;
		m_sign = -m_sign;
		return *this;
	}
	int i;
	bool reverse;
	int maxdigitnum;
	if (m_digitnum < in.m_digitnum)
	{
		reverse = true;
		maxdigitnum = in.m_digitnum;
	}
	else if (m_digitnum > in.m_digitnum)
	{
		reverse = false;
		maxdigitnum = m_digitnum;
	}
	else
	{
		for (i = m_digitnum - 1; i >= 0; i--)
		{
			if (m_digits[i] > in.m_digits[i])
			{
				reverse = false;
				break;
			}
			else if (m_digits[i] < in.m_digits[i])
			{
				reverse = true;
				break;
			}
		}
		maxdigitnum = i + 1;
		if (maxdigitnum == 0)
		{
			m_digitnum = 0;
			m_sign = 1;
			return *this;
		}
	}
	if (reverse)
	{
		if (maxdigitnum >= m_maxdigitnum)
		{
			// 桁数拡張あり
			int newdigitnum = NextMemSize(maxdigitnum + 1);
			int* p = MM_MALLOC(int, newdigitnum);
			int c = 0;
			for (i = 0; i < m_digitnum; i++)
			{
				p[i] = BIGINT_BASE + in.m_digits[i] - m_digits[i] - c;
				c = 1 - p[i] / BIGINT_BASE;
				p[i] %= BIGINT_BASE;
			}
			for (; i < maxdigitnum && c; i++)
			{
				p[i] = BIGINT_BASE + in.m_digits[i] - c;
				c = 1 - p[i] / BIGINT_BASE;
				p[i] %= BIGINT_BASE;
			}
			if (i < maxdigitnum)
			{
				memcpy(p + i, in.m_digits + i, sizeof(int)* (maxdigitnum - i));
			}
			m_digitnum = 0;
			for (i = maxdigitnum - 1; i >= 0; i--)
			{
				if (p[i])
				{
					m_digitnum = i + 1;
					break;
				}
			}
			MM_FREE(m_digits);
			m_digits = p;
			m_maxdigitnum = newdigitnum;
		}
		else
		{
			int c = 0;
			for (i = 0; i < m_digitnum; i++)
			{
				m_digits[i] = BIGINT_BASE + in.m_digits[i] - m_digits[i] - c;
				c = 1 - m_digits[i] / BIGINT_BASE;
				m_digits[i] %= BIGINT_BASE;
			}
			for (; i < maxdigitnum && c; i++)
			{
				m_digits[i] = BIGINT_BASE + in.m_digits[i] - c;
				c = 1 - m_digits[i] / BIGINT_BASE;
				m_digits[i] %= BIGINT_BASE;
			}
			if (i < maxdigitnum)
			{
				memcpy(m_digits + i, in.m_digits + i, sizeof(int)* (maxdigitnum - i));
			}
			m_digitnum = 0;
			for (i = maxdigitnum - 1; i >= 0; i--)
			{
				if (m_digits[i])
				{
					m_digitnum = i + 1;
					break;
				}
			}
		}
		m_sign = -m_sign;
	}
	else
	{
		int c = 0;
		for (i = 0; i < in.m_digitnum; i++)
		{
			m_digits[i] += BIGINT_BASE - in.m_digits[i] - c;
			c = 1 - m_digits[i] / BIGINT_BASE;
			m_digits[i] %= BIGINT_BASE;
		}
		for (; i < maxdigitnum && c; i++)
		{
			m_digits[i] += BIGINT_BASE - c;
			c = 1 - m_digits[i] / BIGINT_BASE;
			m_digits[i] %= BIGINT_BASE;
		}
		m_digitnum = 0;
		for (i = maxdigitnum - 1; i >= 0; i--)
		{
			if (m_digits[i])
			{
				m_digitnum = i + 1;
				break;
			}
		}
	}
	return *this;
}

CBigInt& CBigInt::operator*=(const CBigInt& in)
{
	if (m_digitnum == 0)
	{
		return *this;
	}
	if (in.m_digitnum == 0)
	{
		*this = 0;
		return *this;
	}
	else if (in.m_digitnum < 4)
	{
		m_sign *= in.m_sign;
		// 3桁未満の掛け算
		if (in.m_digitnum == 1 && in.m_digits[0] == 1)
		{
		}
		else if (m_digitnum + in.m_digitnum > m_maxdigitnum)
		{
			// 桁数拡張あり
			long long n = (unsigned long long)in;
			if (n < 0)
			{
				n = -n;
			}
			int digit = NextMemSize(m_digitnum + in.m_digitnum);
			int* p = MM_MALLOC(int, digit);
			int i;
			long long c = 0;
			for (i = 0; i < m_digitnum; i++)
			{
				c += m_digits[i] * n;
				p[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			for (; c; i++)
			{
				p[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			m_digitnum = i;
			MM_FREE(m_digits);
			m_digits = p;
			m_maxdigitnum = digit;
		}
		else
		{
			// 桁数拡張無し
			long long n = (unsigned long long)in;
			if (n < 0)
			{
				n = -n;
			}
			int i;
			long long c = 0;
			for (i = 0; i < m_digitnum; i++)
			{
				c += m_digits[i] * n;
				m_digits[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			for (; c; i++)
			{
				m_digits[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			m_digitnum = i;
		}
	}
	else if (m_digitnum < 4)
	{
		if (m_digitnum + in.m_digitnum > m_maxdigitnum)
		{
			m_sign *= in.m_sign;
			long long n = (unsigned long long)*this;
			int digit = NextMemSize(m_digitnum + in.m_digitnum);
			int* p = MM_MALLOC(int, digit);
			int i;
			long long c = 0;
			for (i = 0; i < in.m_digitnum; i++)
			{
				c += in.m_digits[i] * n;
				p[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			for (; c; i++)
			{
				p[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			m_digitnum = i;
			MM_FREE(m_digits);
			m_digits = p;
			m_maxdigitnum = digit;
		}
		else
		{
			m_sign *= in.m_sign;
			long long n = (unsigned long long)*this;
			int i;
			long long c = 0;
			for (i = 0; i < in.m_digitnum; i++)
			{
				c += in.m_digits[i] * n;
				m_digits[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			for (; c; i++)
			{
				m_digits[i] = c % BIGINT_BASE;
				c /= BIGINT_BASE;
			}
			m_digitnum = i;
		}
	}
	else
	{
		if (std::min(in.m_digitnum, m_digitnum) <= 32)
		{
			*this = std::move(KaratsubaMul(*this, in));
		}
		else if(std::max(in.m_digitnum, m_digitnum) > 64)
		{
			*this = std::move(FFTMul(*this, in));
		}
		else
		{
			CBigInt bak = *this;
			CBigInt kar = KaratsubaMul(*this, in);
			int divide = std::max(m_digitnum, in.m_digitnum) / 2;

			CBigInt a, b, c, d, ac, bd, mida, midb;
			a = *this;
			a >>= divide;
			b = *this;
			b.Clip(divide);
			c = in;
			c >>= divide;
			d = in;
			d.Clip(divide);
			mida = a;
			mida -= b;
			midb = d;
			midb -= c;
			mida *= midb;
			a *= c;
			b *= d;
			mida += a;
			mida += b;
			mida <<= divide;
			*this = a;
			*this <<= divide * 2;
			*this += b;
			*this += mida;
		}
	}
	return *this;
}

CBigInt& CBigInt::operator/=(const CBigInt& in)
{
	if (in.m_digitnum == 0)
	{
		printf("divide by zero\n");
		abort();
	}
	else if (in.m_digitnum < 4)
	{
		m_sign *= in.m_sign;
		// 3桁未満の割り算
		if (in.m_digitnum == 1 && in.m_digits[0] == 1)
		{
		}
		else
		{
			long long n = (unsigned long long)in;
			if (n < 0)
			{
				n = -n;
			}
			int i;
			long long c = 0;
			int digitnum = 0;
			for (i = m_digitnum - 1; i >= 0; i--)
			{
				c += m_digits[i];
				m_digits[i] = c / n;
				c -= m_digits[i] * n;
				if (digitnum == 0 && m_digits[i] > 0)
				{
					digitnum = i + 1;
				}
				c *= BIGINT_BASE;
			}
			m_digitnum = digitnum;
		}
	}
	else
	{
		CBigInt posin = in;
		posin.m_sign = 1;
		int sign = m_sign * in.m_sign;
		m_sign = 1;
		if (posin > *this)
		{
			m_sign = 1;
			m_digitnum = 0;
			return *this;
		}
		int digits = m_digitnum - posin.m_digitnum;
		if (digits < 0)
		{
			// バグ
			abort();
		}
		int shift = posin.m_digitnum + digits + 2;
		int shift2 = m_digitnum - digits - 2;
		if (shift2 < 0)
		{
			shift2 = 0;
		}
		CBigInt inv = posin;
		inv.Inverse(shift);
		CBigInt div = *this;
		if (shift2)
		{
			div >>= shift2;
		}
		div *= inv;
		div >>= shift - shift2;
#if 1 // 除算に誤差を許す
		CBigInt tmp = posin * div;
		if (*this < tmp)
		{
			while (*this < tmp)
			{
				div -= 1;
				tmp -= posin;
			}
		}
		else
		{
			while (*this >= tmp)
			{
				div += 1;
				tmp += posin;
			}
			div -= 1;
		}
#endif
		div.m_sign = sign;
		*this = div;
	}
	return *this;
}

CBigInt& CBigInt::operator%=(const CBigInt& in)
{
	*this = *this - *this / in * in;
	return *this;
}

CBigInt& CBigInt::operator^=(long long n)
{
	*this = *this ^ n;
	return *this;
}

CBigInt CBigInt::operator^(long long n)
{
	long long a = 1;
	while (a <= n)
	{
		a <<= 1;
	}
	a >>= 1;
	CBigInt ans;
	ans = *this;
	while (a > 1)
	{
		a >>= 1;
		ans *= ans;
		if (n & a)
		{
			ans *= *this;
		}
	}
	return ans;
}

CBigInt CBigInt::operator+(const CBigInt& in) const
{
	CBigInt ret = *this;
	return std::move(ret += in);
}
CBigInt CBigInt::operator-(const CBigInt& in) const
{
	CBigInt ret = *this;
	return std::move(ret -= in);
}
CBigInt CBigInt::operator*(const CBigInt& in) const
{
	CBigInt ret = *this;
	return std::move(ret *= in);
}
CBigInt CBigInt::operator/(const CBigInt& in) const
{
	CBigInt ret = *this;
	return std::move(ret /= in);
}
CBigInt CBigInt::operator%(const CBigInt& in) const
{
	return std::move(*this - ((*this / in) *= in));
}

CBigInt CBigInt::operator+(long long n) const
{
	CBigInt ret = *this;
	return std::move(ret += n);
}
CBigInt CBigInt::operator-(long long n) const
{
	CBigInt ret = *this;
	return std::move(ret -= n);
}
CBigInt CBigInt::operator*(long long n) const
{
	CBigInt ret = *this;
	return std::move(ret *= n);
}
CBigInt CBigInt::operator/(long long n) const
{
	CBigInt ret = *this;
	return std::move(ret /= n);
}

long long CBigInt::operator%(long long n) const
{
	int sign = 1;
	if (n < 0)
	{
		sign = -1;
		n = -n;
	}
	if (n > (1ll << 62) / BIGINT_BASE)
	{
		CBigInt ret = *this;
		return (long long)(ret %= n);
	}
	else
	{
		int i;
		long long c = 0;
		int digitnum = 0;
		for (i = m_digitnum - 1; i >= 0; i--)
		{
			c *= BIGINT_BASE;
			c += m_digits[i];
			long long t = c / n;
			c -= t * n;
		}
		return c;
	}
}

CBigInt CBigInt::operator<<(int n) const
{
	CBigInt ret = *this;
	return std::move(ret <<= n);
}
CBigInt CBigInt::operator>>(int n) const
{
	CBigInt ret = *this;
	return std::move(ret >>= n);
}

CBigInt& CBigInt::operator<<=(int n)
{
	if (n == 0 || m_digitnum == 0)
	{
		return *this;
	}
	else if (n < 0)
	{
		return *this >>= -n;
	}
	int digit = n + m_digitnum;
	if (digit > m_maxdigitnum)
	{
		int maxdigit = NextMemSize(digit);
		int* p = MM_MALLOC(int, maxdigit);
		memcpy(p + n, m_digits, sizeof(int) * m_digitnum);
		memset(p, 0, sizeof(int) * n);
		MM_FREE(m_digits);
		m_digits = p;
		m_digitnum = digit;
		m_maxdigitnum = maxdigit;
	}
	else
	{
		int i;
		for (i = m_digitnum - 1; i >= 0; i--)
		{
			m_digits[i + n] = m_digits[i];
		}
		memset(m_digits, 0, sizeof(int) * n);
		m_digitnum = digit;
	}
	return *this;
}

CBigInt& CBigInt::operator>>=(int n)
{
	if (n == 0 || m_digitnum == 0)
	{
		return *this;
	}
	else if (n < 0)
	{
		return *this <<= -n;
	}
	if (m_digitnum <= n)
	{
		m_digitnum = 0;
		m_sign = 1;
	}
	else
	{
		int i;
		for (i = 0; i < m_digitnum - n; i++)
		{
			m_digits[i] = m_digits[i + n];
		}
		m_digitnum -= n;
	}
	return *this;
}

CBigInt CBigInt::operator-() const
{
	CBigInt ret = *this;
	ret.m_sign *= -1;
	return ret;
}

bool CBigInt::operator>(const CBigInt& in) const
{
	if (m_digitnum + in.m_digitnum == 0)
	{
		return false;
	}
	if (m_sign != in.m_sign)
	{
		return m_sign > in.m_sign;
	}
	else if (m_digitnum != in.m_digitnum)
	{
		return m_digitnum * m_sign > in.m_digitnum * m_sign;
	}
	else
	{
		int i;
		for (i = m_digitnum - 1; i >= 0; i--)
		{
			if (m_digits[i] != in.m_digits[i])
			{
				return m_digits[i] * m_sign > in.m_digits[i] * m_sign;
			}
		}
		return false;
	}
}

bool CBigInt::operator<(const CBigInt& in) const
{
	return in > *this;
}

bool CBigInt::operator>=(const CBigInt& in) const
{
	return !(in > *this);
}

bool CBigInt::operator<=(const CBigInt& in) const
{
	return !(*this > in);
}

bool CBigInt::operator==(const CBigInt& in) const
{
	if (m_digitnum + in.m_digitnum == 0)
	{
		return true;
	}
	if (m_sign != in.m_sign)
	{
		return false;
	}
	else if (m_digitnum != in.m_digitnum)
	{
		return false;
	}
	else
	{
		int i;
		for (i = m_digitnum - 1; i >= 0; i--)
		{
			if (m_digits[i] != in.m_digits[i])
			{
				return false;
			}
		}
		return true;
	}
}

bool CBigInt::operator!=(const CBigInt& in) const
{
	return !(*this == in);
}

CBigInt::operator int() const
{
	int ret = 0;
	int i;
	for (i = m_digitnum - 1; i >= 0; i--)
	{
		ret *= BIGINT_BASE;
		ret += m_digits[i];
	}
	return ret * m_sign;
}

CBigInt::operator long long() const
{
	long long ret = 0;
	int i;
	for (i = m_digitnum - 1; i >= 0; i--)
	{
		ret *= BIGINT_BASE;
		ret += m_digits[i];
	}
	return ret * m_sign;
}

CBigInt::operator unsigned long long() const
{
	unsigned long long ret = 0;
	int i;
	for (i = m_digitnum - 1; i >= 0; i--)
	{
		ret *= BIGINT_BASE;
		ret += m_digits[i];
	}
	return ret;
}

CBigInt::operator double() const
{
	double ret = 0;
	int i;
	for (i = m_digitnum - 1; i >= 0; i--)
	{
		ret *= BIGINT_BASE;
		ret += m_digits[i];
	}
	return ret * m_sign;
}



CBigInt& CBigInt::ShiftAdd(const CBigInt& in, int shift)
{
	printf("ShiftAddは未実装です");
	return *this;
}

CBigInt& CBigInt::ShiftSub(const CBigInt& in, int shift)
{
	printf("ShiftSubは未実装です");
	return *this;
}

CBigInt& CBigInt::Clip(int n)
{
	int i;
	if (n >= m_digitnum)
	{
		return *this;
	}
	if (n <= 0)
	{
		m_digitnum = 0;
		m_sign = 1;
		return *this;
	}
	for (i = n - 1; i >= 0 && m_digits[i] == 0; i--);
	m_digitnum = i + 1;
	if (m_digitnum == 0)
	{
		m_sign = 1;
	}
	return *this;
}

CBigInt& CBigInt::Inverse(int digit)
{
	CBigInt inv;
	int d = digit - m_digitnum;
	if (d < 0)
	{
		*this = 0;
		return *this;
	}
	CBigInt n = *this;
	if (d + 2 < n.m_digitnum)
	{
		int shift1 = n.m_digitnum - (d + 2);
		n >>= shift1;
		digit -= shift1;
	}
	if (d > 16)
	{
		int shift2 = digit - d / 2 + 2;
		inv = n;
		inv.Inverse(shift2);
		// x = x(2 - a * x)
		CBigInt two = 2;
		two <<= shift2;
		inv = inv * (two - (n * inv));
		inv >>= shift2 * 2 - digit;
	}
	else
	{
		double dinv = 1 / (double)n;
		int i;
		CBigInt two = 2;
		two <<= digit;
		for (i = 0; i < digit; i++)
		{
			dinv *= BIGINT_BASE;
		}
		inv = dinv;
		for (i = 0; i < 4; i++)
		{
			inv = inv * (two - (n * inv));
			inv >>= digit;
		}
	}
	*this = inv;
	return *this;
}

CBigInt& CBigInt::ModPow(int x, int maxdigit)
{
	int a = 1;
	while (a <= x)
	{
		a <<= 1;
	}
	a >>= 1;
	CBigInt ans;
	ans = *this;
	while (a > 1)
	{
		a >>= 1;
		ans *= ans;
		ans.Clip(maxdigit);
		if (x & a)
		{
			ans *= *this;
			ans.Clip(maxdigit);
		}
	}
	*this = std::move(ans);
	return *this;
}

CBigInt ModPow(const CBigInt& a, CBigInt x, const CBigInt& M)
{
	CBigInt ans = 1;
	CBigInt pow2;
	pow2 = a;
	while (x > 0)
	{
		if (x.m_digits[0] & 1)
		{
			ans *= pow2;
			ans %= M;
		}
		x /= 2;
		if (x > 0)
		{
			pow2 *= pow2;
			pow2 %= M;
		}
	}
	return ans;
}
int CBigInt::GetDigit10(int n)
{
	int na = n / BIGINT_BASE_DIGITS;
	int nb = n % BIGINT_BASE_DIGITS;
	int i;
	int a = m_digits[na];
	for (i = 0; i < nb; i++)
	{
		a /= 10;
	}
	return a % 10;
}

int CBigInt::GetDigit(int n)
{
	return m_digits[n];
}

CBigInt GCD(CBigInt a, CBigInt b)
{
	while (b > 0)
	{
		a = a % b;
		CBigInt t = std::move(a);
		a = std::move(b);
		b = std::move(t);
	}
	return a;
}

void CBigInt::Print(FILE* fout) const
{
	int i;
	if (fout == NULL)
	{
		fout = stdout;
	}
	if (m_digitnum == 0)
	{
		fputs("0", fout);
	}
	else
	{
		if (m_sign == -1)
		{
			fputs("-", fout);
		}
		else
		{
			//fputs(" ", fout);
		}
		fprintf(fout, BIGINT_FORMAT_TOP, m_digits[m_digitnum - 1]);
		for (i = m_digitnum - 2; i >= 0; i--)
		{
			fprintf(fout, BIGINT_FORMAT, m_digits[i]);
		}
	}
}

void CBigInt::PrintN(FILE* fout) const
{
	//	printf("%8d digits:", m_digitnum);
	Print(fout);
	if (fout == NULL)
	{
		fout = stdout;
	}
	fputs("\n", fout);
}

char* CBigInt::SPrint(char* s) const
{
	int i;
	char * p = s;
	if (m_digitnum == 0)
	{
		p += sprintf(p, "0");
	}
	else
	{
		if (m_sign == -1)
		{
			p += sprintf(p, "-");
		}
		else
		{
		}
		p += sprintf(p, BIGINT_FORMAT_TOP, m_digits[m_digitnum - 1]);
		for (i = m_digitnum - 2; i >= 0; i--)
		{
			p += sprintf(p, BIGINT_FORMAT, m_digits[i]);
		}
	}
	*p = 0;
	return s;
}
