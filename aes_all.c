#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <immintrin.h>

#define MAX_DATA_SIZE (32 * 1024 * 1024)
#define u32 uint32_t

void padding(uint8_t *data, uint32_t padding_len)
{
    uint8_t padding_content = (uint8_t)padding_len;
    for (uint32_t i = 0; i <= padding_len; i++)
        data[16 - i] = padding_content;
}

void key_expand_simd(uint8_t *key, __m128i *expand_key)
{
    __m128i work = _mm_loadu_si128((const __m128i *)(key));
    expand_key[0] = work;

    __m128i t, t2;

#define EXPAND(n, Rcon)                           \
    t = _mm_slli_si128(work, 4);                  \
    t = _mm_xor_si128(work, t);                   \
    t2 = _mm_slli_si128(t, 8);                    \
    t = _mm_xor_si128(t, t2);                     \
    work = _mm_aeskeygenassist_si128(work, Rcon); \
    work = _mm_shuffle_epi32(work, 0xFF);         \
    work = _mm_xor_si128(t, work);                \
    expand_key[n] = work;

    EXPAND(1, 0x01);
    EXPAND(2, 0x02);
    EXPAND(3, 0x04);
    EXPAND(4, 0x08);
    EXPAND(5, 0x10);
    EXPAND(6, 0x20);
    EXPAND(7, 0x40);
    EXPAND(8, 0x80);
    EXPAND(9, 0x1B);
    EXPAND(10, 0x36);
}

void inv_key_expand_simd(uint8_t *key, __m128i *expand_key)
{
    key_expand_simd(key, expand_key);

    for (int i = 1; i <= 9; i++)
        expand_key[i] = _mm_aesimc_si128(expand_key[i]);
}

void encrypt_simd(__m128i *data, __m128i *expand_key, __m128i *res)
{
    __m128i state = _mm_xor_si128(*data, expand_key[0]);

    for (int i = 1; i <= 9; i++)
        state = _mm_aesenc_si128(state, expand_key[i]);

    state = _mm_aesenclast_si128(state, expand_key[10]);

    _mm_storeu_si128(res, state);
}

void eq_decrypt_simd(__m128i *data, __m128i *expand_key, __m128i *res)
{
    __m128i state = _mm_xor_si128(*data, expand_key[10]);

    for (int i = 9; i >= 1; i--)
        state = _mm_aesdec_si128(state, expand_key[i]);

    state = _mm_aesdeclast_si128(state, expand_key[0]);

    _mm_storeu_si128(res, state);
}

void ecb_encrypt(__m128i *expand_key, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16];
    __m128i res;

    uint32_t i = 0;
    while (i < len)
    {
        if (i + 16 > len)
        {
            fread(data, sizeof(uint8_t), len - i, in);
            padding(data, 16 - (len - i));
        }
        else
            fread(data, sizeof(uint8_t), 16, in);

        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));
        encrypt_simd(&data_, expand_key, &res);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), 16, out);
        i += 16;
    }

    if (i == len)
    {
        padding(data, 16);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));
        encrypt_simd(&data_, expand_key, &res);
        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), 16, out);
    }
}

void ecb_decrypt(__m128i *expand_key, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16];
    __m128i res;

    uint32_t i = 0;
    while (i < len)
    {
        fread(data, sizeof(uint8_t), 16, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        eq_decrypt_simd(&data_, expand_key, &res);

        int len_ = 16;
        if (i + 16 == len)
        {
            uint8_t *res_ = (uint8_t *)&res;
            len_ = 16 - (int)res_[15];
        }

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);

        i += 16;
    }
}

void cbc_encrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16];
    __m128i last = _mm_loadu_si128((const __m128i *)(IV));
    __m128i res;

    uint32_t i = 0;
    while (i < len)
    {
        if (i + 16 > len)
        {
            fread(data, sizeof(uint8_t), len - i, in);
            padding(data, 16 - (len - i));
        }
        else
            fread(data, sizeof(uint8_t), 16, in);

        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));
        data_ = _mm_xor_si128(data_, last);
        encrypt_simd(&data_, expand_key, &res);
        last = res;

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), 16, out);
        i += 16;
    }

    if (i == len)
    {
        padding(data, 16);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));
        data_ = _mm_xor_si128(data_, last);
        encrypt_simd(&data_, expand_key, &res);
        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), 16, out);
    }
}

void cbc_decrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16];
    __m128i last = _mm_loadu_si128((const __m128i *)(IV));
    __m128i res;

    uint32_t i = 0;
    while (i < len)
    {
        fread(data, sizeof(uint8_t), 16, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        eq_decrypt_simd(&data_, expand_key, &res);
        res = _mm_xor_si128(res, last);

        int len_ = 16;
        if (i + 16 == len)
        {
            uint8_t *res_ = (uint8_t *)&res;
            len_ = 16 - (int)res_[15];
        }

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);

        i += 16;
        last = data_;
    }
}

void cfb_encrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16] = {0};
    __m128i z = _mm_loadu_si128((const __m128i *)(IV));
    __m128i res;

    for (uint32_t i = 0; i < len; i += 16)
    {
        encrypt_simd(&z, expand_key, &res);

        int len_ = 16;
        if (i + 16 > len)
            len_ = len - i;

        fread(data, sizeof(uint8_t), len_, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        res = _mm_xor_si128(res, data_);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);

        z = res;
    }
}

void cfb_decrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16] = {0};
    __m128i z = _mm_loadu_si128((const __m128i *)(IV));
    __m128i res;

    for (uint32_t i = 0; i < len; i += 16)
    {
        encrypt_simd(&z, expand_key, &res);

        int len_ = 16;
        if (i + 16 > len)
            len_ = len - i;
        fread(data, sizeof(uint8_t), len_, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        res = _mm_xor_si128(res, data_);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);

        z = data_;
    }
}

void ofb_encrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16] = {0};
    __m128i z = _mm_loadu_si128((const __m128i *)(IV));
    __m128i res;

    for (uint32_t i = 0; i < len; i += 16)
    {
        encrypt_simd(&z, expand_key, &res);

        z = res;

        int len_ = 16;
        if (i + 16 > len)
            len_ = len - i;

        fread(data, sizeof(uint8_t), len_, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        res = _mm_xor_si128(res, data_);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);
    }
}

void ofb_decrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16] = {0};
    __m128i z = _mm_loadu_si128((const __m128i *)(IV));
    __m128i res;

    for (uint32_t i = 0; i < len; i += 16)
    {
        encrypt_simd(&z, expand_key, &res);

        z = res;

        int len_ = 16;
        if (i + 16 > len)
            len_ = len - i;
        fread(data, sizeof(uint8_t), len_, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        res = _mm_xor_si128(res, data_);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);
    }
}

void ctr_encrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16] = {0};
    __m128i ctr = _mm_loadu_si128((const __m128i *)IV);
    __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    ctr = _mm_shuffle_epi8(ctr, mask);

    __m128i res;

    for (uint32_t i = 0; i < len; i += 16)
    {
        __m128i T = _mm_shuffle_epi8(ctr, mask);

        encrypt_simd(&T, expand_key, &res);

        int len_ = 16;
        if (i + 16 > len)
            len_ = len - i;

        fread(data, sizeof(uint8_t), len_, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        res = _mm_xor_si128(res, data_);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);

        ctr = _mm_add_epi64(ctr, _mm_set_epi64x(0, 1));
    }
}

void ctr_decrypt(__m128i *expand_key, uint8_t *IV, uint32_t len, FILE *in, FILE *out)
{
    uint8_t data[16] = {0};
    __m128i ctr = _mm_loadu_si128((const __m128i *)IV);
    __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    ctr = _mm_shuffle_epi8(ctr, mask);

    __m128i res;

    for (uint32_t i = 0; i < len; i += 16)
    {
        __m128i T = _mm_shuffle_epi8(ctr, mask);

        encrypt_simd(&T, expand_key, &res);

        int len_ = 16;
        if (i + 16 > len)
            len_ = len - i;

        fread(data, sizeof(uint8_t), len_, in);
        __m128i data_ = _mm_loadu_si128((const __m128i *)(data));

        res = _mm_xor_si128(res, data_);

        uint8_t *res_ = (uint8_t *)&res;
        fwrite(res_, sizeof(uint8_t), len_, out);

        ctr = _mm_add_epi64(ctr, _mm_set_epi64x(0, 1));
    }
}

int main(void)
{
#ifndef ONLINE_JUDGE
    FILE *in = fopen("aes_ctr_i2.bin", "rb");
    FILE *out = fopen("aeso.bin", "wb");
#else
#define in stdin
#define out stdout
#endif

    uint8_t mode;
    uint8_t key[16];
    uint8_t IV[16];
    uint32_t len;

    fread(&mode, sizeof(uint8_t), 1, in);
    fread(key, sizeof(uint8_t), 16, in);
    fread(IV, sizeof(uint8_t), 16, in);
    fread(&len, sizeof(uint32_t), 1, in);

    __m128i expand_key[11];
    if (mode == 0x80 || mode == 0x81)
        inv_key_expand_simd(key, expand_key);
    else
        key_expand_simd(key, expand_key);

    switch (mode)
    {
    case 0x00:
        ecb_encrypt(expand_key, len, in, out);
        break;
    case 0x01:
        cbc_encrypt(expand_key, IV, len, in, out);
        break;
    case 0x02:
        cfb_encrypt(expand_key, IV, len, in, out);
        break;
    case 0x03:
        ofb_encrypt(expand_key, IV, len, in, out);
        break;
    case 0x04:
        ctr_encrypt(expand_key, IV, len, in, out);
        break;
    case 0x80:
        ecb_decrypt(expand_key, len, in, out);
        break;
    case 0x81:
        cbc_decrypt(expand_key, IV, len, in, out);
        break;
    case 0x82:
        cfb_decrypt(expand_key, IV, len, in, out);
        break;
    case 0x83:
        ofb_decrypt(expand_key, IV, len, in, out);
        break;
    case 0x84:
        ctr_decrypt(expand_key, IV, len, in, out);
        break;
    }

    return 0;
}