#pragma once

#include <iostream>
#include <stdio.h>

class CBigInt;

#if 1
#define MM_MALLOC(type, size) ((type*)(AlignedMalloc(sizeof(type) * size)))
#define MM_FREE(var) AlignedFree(var)
#else
#define MM_MALLOC(type, size) ((type*)(_mm_malloc(sizeof(type) * size, 32)))
#define MM_FREE(var) _mm_free(var)
#endif

void BigIntInit(int size);
void* AlignedMalloc(size_t size);
void AlignedFree(void* ptr);
int NextPow2(int n);
int NextMemSize(int n);
CBigInt ModPow(const CBigInt& a, CBigInt x, const CBigInt& M);
CBigInt GCD(CBigInt a, CBigInt b);

#define BIGINT_BASE 10000
#define BIGINT_BASE_DIGITS 4
#define BIGINT_FORMAT "%04d"
#define BIGINT_FORMAT_TOP "%d"
class CBigInt
{
public:
	int m_maxdigitnum;
	int m_digitnum;
	int* m_digits;
	int m_sign;

	CBigInt();
	CBigInt(const CBigInt& in);
	CBigInt(CBigInt&& in);
	CBigInt(char* in) {
		*this = in;
	};
	CBigInt(const char* in);
	template <class T> CBigInt(T in);
	~CBigInt();
	CBigInt& operator= (char* in) {
		this->m_digits = NULL;
		this->m_digitnum = 0;
		return *this = (const char*)in;
	};
	CBigInt& operator= (const char* in);
	template <class T> CBigInt& operator= (T in);
#ifdef _MSC_VER
	template <> CBigInt& operator= (double in);
#else
	CBigInt& operator= (double in);
#endif
	CBigInt& operator= (const CBigInt& in);
	CBigInt& operator= (CBigInt&& in);

	CBigInt& operator+=(const CBigInt& in);
	CBigInt& operator-=(const CBigInt& in);
	CBigInt& operator*=(const CBigInt& in);
	CBigInt& operator/=(const CBigInt& in);
	CBigInt& operator%=(const CBigInt& in);
	CBigInt& operator^=(long long n);
	CBigInt& operator<<=(int n);
	CBigInt& operator>>=(int n);

	CBigInt operator+(const CBigInt& in) const;
	CBigInt operator-(const CBigInt& in) const;
	CBigInt operator*(const CBigInt& in) const;
	CBigInt operator/(const CBigInt& in) const;
	CBigInt operator%(const CBigInt& in) const;
	CBigInt operator+(long long n) const;
	CBigInt operator-(long long n) const;
	CBigInt operator*(long long n) const;
	CBigInt operator/(long long n) const;
	long long operator%(long long n) const;
	CBigInt operator^(long long n);
	CBigInt operator<<(int n) const;
	CBigInt operator>>(int n) const;

	CBigInt operator-() const;
	bool operator>(const CBigInt& in) const;
	bool operator<(const CBigInt& in) const;
	bool operator>=(const CBigInt& in) const;
	bool operator<=(const CBigInt& in) const;
	bool operator==(const CBigInt& in) const;
	bool operator!=(const CBigInt& in) const;

	explicit operator int() const;
	explicit operator long long() const;
	explicit operator unsigned long long() const;
	explicit operator double() const;

	CBigInt& ShiftAdd(const CBigInt& in, int shift);
	CBigInt& ShiftSub(const CBigInt& in, int shift);

	CBigInt& Clip(int n);
	CBigInt& Inverse(int digit);
	CBigInt& ModPow(int x, int maxdigit);
	int GetDigit(int n);
	int GetDigit10(int n);

	void Print(FILE* fout = NULL) const;
	void PrintN(FILE* fout = NULL) const;
	char* SPrint(char* s) const;
};
