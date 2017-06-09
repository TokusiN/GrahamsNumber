#include "BigInt.h"

#include <time.h>
#include <iostream>
#include <stdio.h>


CBigInt ClipPow3(CBigInt x, long long mod)
{
	CBigInt a = 1;
	CBigInt pow2;
	pow2 = 3;
	while (x > 0)
	{
		if (x.m_digits[0] & 1)
		{
			a *= pow2;
			a.Clip(mod / 4);
		}
		x /= 2;
		if (x > 0)
		{
			pow2 *= pow2;
			pow2.Clip(mod / 4);
		}
	}
	return std::move(a);
}

CBigInt ModPow3(const CBigInt& x, long long mod, CBigInt* laste = NULL)
{
	long long k = mod - 1;
	long long b = 0;
	while (k > 100000000ll || k * k > mod)
	{
		k /= 2;
		b++;
	}
	k = (k - 1 | 3) + 1;
	int i;
	CBigInt a = ClipPow3(CBigInt(1) << (k / 4), mod);
	CBigInt e = a;
	e >>= (k / 4);
	CBigInt result = x;
	result = ClipPow3(result.Clip(k / 4), mod);
	CBigInt p = 1;
	p <<= (k / 4);
	for (; b; b--)
	{
		CBigInt e1 = e;
		CBigInt M;

		CBigInt nextresult = 1;
		CBigInt xrange = x;
		xrange >>= k / 4;
		xrange.Clip(k / 4);
		M = 1;
		for (i = 0; i < 1 << b; i++)
		{
			M *= (xrange - i);
			M /= i + 1;
			CBigInt tmp = (e * M).Clip((mod - (i + 1) * k) / 4) << (i + 1) * (k / 4);
			nextresult += tmp;
			e *= e1;
			e.Clip((mod - i*k) / 4);
		}
		result *= nextresult;
		result.Clip(mod / 4);

		if (b > 1)
		{
			e = e1;
			CBigInt nexte = 1;
			M = 1;
			for (i = 0; i < 1 << b; i++)
			{
				M *= (p - i);
				M /= i + 1;
				CBigInt tmp = (e * M).Clip((mod - (i + 1) * k) / 4) << (i + 1) * (k / 4);
				nexte += tmp;
				e *= e1;
				e.Clip((mod - i*k) / 4);
			}
			nexte.Clip(mod / 4);
			p <<= k / 4;
			k *= 2;
			e = nexte;
			e >>= (k / 4);
		}
		else if (laste)
		{
			*laste = e1;
		}
	}
	result.Clip(mod / 4);
	return std::move(result);

}

CBigInt G(long long digit)
{
	if (digit < 100)
	{
		int digitbase = digit / 4;
		CBigInt a, x;
		a = 3;
		x = 3;
		int start = time(NULL);
		x.ModPow(7, digitbase);
		int n;
		for (n = 1; n < digit; n++)
		{
			a.ModPow(10, digitbase);
			int d = x.GetDigit10(n);
			if (d > 0)
			{
				CBigInt b;
				b = a;
				b.ModPow(d, digitbase);
				x *= b;
				x.Clip(digitbase);
			}
		}
		return x;
	}
	else
	{
		long long halfdigit = ((digit - 1) / 2 | 3) + 1;
		CBigInt g2 = G(halfdigit);
		CBigInt e;
		g2 = ModPow3(g2, digit, &e);
		e *= g2;
		CBigInt e2;
		e2 = e;
		e += 1;
		while (e2.m_digitnum > 0)
		{
			e2 *= e2;
			e2.Clip(halfdigit / 4);
			e += e * e2;
		}
		e -= 1;
		e.Clip(halfdigit / 4);
		CBigInt tmp = (g2 >> (halfdigit / 4)) * e << halfdigit / 4;
		g2 += (g2 >> (halfdigit / 4)) * e << halfdigit / 4;
		g2.Clip(digit / 4);
		return g2;
	}
}

int main()
{
	char* str = 0;

	int DIGITS = 256;
	for (;; DIGITS *= 2)
	{
		if (str)
		{
			delete[] str;
		}
		str = new char[DIGITS + 1];
		int start;
		start = clock();
		CBigInt g = G(DIGITS);

		g.SPrint(str);
		char fn[256];
		sprintf(fn, "%08d.txt", DIGITS);
		FILE* fp = fopen(fn, "w");
		fprintf(fp, "%s\n", str);
		fclose(fp);
		printf("%8d digits :%8dms\n", DIGITS, clock() - start);
	}
	return 0;
}

