#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <immintrin.h>

const uint32_t RCON[11][4] = {{0x00, 0x00, 0x00, 0x00}, {0x01, 0x00, 0x00, 0x00}, {0x02, 0x00, 0x00, 0x00}, {0x04, 0x00, 0x00, 0x00}, {0x08, 0x00, 0x00, 0x00}, {0x10, 0x00, 0x00, 0x00}, {0x20, 0x00, 0x00, 0x00}, {0x40, 0x00, 0x00, 0x00}, {0x80, 0x00, 0x00, 0x00}, {0x1b, 0x00, 0x00, 0x00}, {0x36, 0x00, 0x00, 0x00}};

const uint8_t SBOX[16][16] = {
    {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
    {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
    {0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15},
    {0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75},
    {0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84},
    {0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf},
    {0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8},
    {0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2},
    {0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73},
    {0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb},
    {0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79},
    {0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08},
    {0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a},
    {0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e},
    {0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf},
    {0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16}};

const uint8_t INV_SBOX[16][16] = {
    {0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb},
    {0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb},
    {0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e},
    {0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25},
    {0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92},
    {0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84},
    {0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06},
    {0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b},
    {0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73},
    {0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e},
    {0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b},
    {0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4},
    {0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f},
    {0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef},
    {0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61},
    {0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d}};

uint32_t table[256];
uint32_t inverse_table[256];

void init_tables()
{
    table[0] = 1; // g^0
    for (int i = 1; i < 255; i++)
    {
        // mul g
        table[i] = (table[i - 1] << 1) ^ table[i - 1];
        if (table[i] & 0x100)
            table[i] ^= 0x11b;
    }
    for (int i = 0; i < 255; i++)
        inverse_table[table[i]] = i;
}

uint32_t gf_mul(uint32_t x, uint32_t y)
{
    if (x == 0 || y == 0)
        return 0;

    return table[(inverse_table[x] + inverse_table[y]) % 255];
}

void padding(uint8_t *data, uint32_t padding_len)
{
    uint8_t padding_content = (uint8_t)padding_len;
    for (uint32_t i = 0; i <= padding_len; i++)
        data[16 - i] = padding_content;
}

void rot_word(uint8_t word[4])
{
    uint8_t temp[4];
    for (int i = 0; i < 4; i++)
        temp[i] = word[(i + 1) % 4];

    for (int i = 0; i < 4; i++)
        word[i] = temp[i];
}

void sub_byte(uint8_t *byte)
{
    int row = *byte >> 4;
    int col = *byte & 0x0f;
    *byte = SBOX[row][col];
}

void sub_word(uint8_t word[4])
{
    for (int i = 0; i < 4; i++)
        sub_byte(&word[i]);
}

void sub_bytes(uint8_t state[4][4])
{
    for (int i = 0; i < 4; i++)
        sub_word(state[i]);
}

void inv_sub_byte(uint8_t *byte)
{
    int row = *byte >> 4;
    int col = *byte & 0x0f;
    *byte = INV_SBOX[row][col];
}

void inv_sub_word(uint8_t word[4])
{
    for (int i = 0; i < 4; i++)
        inv_sub_byte(&word[i]);
}

void inv_sub_bytes(uint8_t state[4][4])
{
    for (int i = 0; i < 4; i++)
        inv_sub_word(state[i]);
}

void rcon(uint8_t word[4], int round)
{
    for (int i = 0; i < 4; i++)
        word[i] ^= RCON[round][i];
}

void key_expand(uint8_t key[4][4], uint8_t expand_key[11][4][4])
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            expand_key[0][i][j] = key[i][j];

    for (int i = 1; i <= 10; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            uint8_t temp[4];
            int p = i * 4 + j;
            for (int k = 0; k < 4; k++)
                temp[k] = expand_key[(p - 1) / 4][k][(p - 1) % 4];

            if ((i * 4 + j) % 4 == 0)
            {
                rot_word(temp);
                sub_word(temp);
                rcon(temp, i);
            }

            for (int k = 0; k < 4; k++)
                expand_key[i][k][j] = temp[k] ^ expand_key[(p - 4) / 4][k][(p - 4) % 4];
        }
    }
}

void key_expand_simd(uint8_t *key, __m128i *expand_key)
{
#define EXPAND(n, Rcon)                           \
    t = _mm_slli_si128(work, 4);                  \
    t = _mm_xor_si128(work, t);                   \
    t2 = _mm_slli_si128(t, 8);                    \
    t = _mm_xor_si128(t, t2);                     \
    work = _mm_aeskeygenassist_si128(work, Rcon); \
    work = _mm_shuffle_epi32(work, 0xFF);         \
    work = _mm_xor_si128(t, work);                \
    expand_key[n] = work;

    __m128i work = _mm_loadu_si128((const __m128i *)(key));
    expand_key[0] = work;
    __m128i t, t2;

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

void add_round_key(uint8_t state[4][4], uint8_t round_key[4][4])
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] ^= round_key[i][j];
}

void shift_row(uint8_t state[4][4])
{
    uint32_t temp[4][4];
    for (int i = 0; i < 4; i++)
    {
        temp[0][i] = state[0][i];
        temp[1][i] = state[1][(i + 1) % 4];
        temp[2][i] = state[2][(i + 2) % 4];
        temp[3][i] = state[3][(i + 3) % 4];
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] = temp[i][j];
}

void inv_shift_row(uint8_t state[4][4])
{
    uint32_t temp[4][4];
    for (int i = 0; i < 4; i++)
    {
        temp[0][i] = state[0][i];
        temp[1][i] = state[1][(i + 3) % 4];
        temp[2][i] = state[2][(i + 2) % 4];
        temp[3][i] = state[3][(i + 1) % 4];
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] = temp[i][j];
}

void mix_column(uint8_t state[4][4])
{
    uint32_t temp[4][4];
    for (int i = 0; i < 4; i++)
    {
        temp[0][i] = gf_mul(0x02, state[0][i]) ^ gf_mul(0x03, state[1][i]) ^ state[2][i] ^ state[3][i];
        temp[1][i] = state[0][i] ^ gf_mul(0x02, state[1][i]) ^ gf_mul(0x03, state[2][i]) ^ state[3][i];
        temp[2][i] = state[0][i] ^ state[1][i] ^ gf_mul(0x02, state[2][i]) ^ gf_mul(0x03, state[3][i]);
        temp[3][i] = gf_mul(0x03, state[0][i]) ^ state[1][i] ^ state[2][i] ^ gf_mul(0x02, state[3][i]);
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] = temp[i][j];
}

void inv_mix_column(uint8_t state[4][4])
{
    uint32_t temp[4][4];
    for (int i = 0; i < 4; i++)
    {
        temp[0][i] = gf_mul(0x0e, state[0][i]) ^ gf_mul(0x0b, state[1][i]) ^ gf_mul(0x0d, state[2][i]) ^ gf_mul(0x09, state[3][i]);
        temp[1][i] = gf_mul(0x09, state[0][i]) ^ gf_mul(0x0e, state[1][i]) ^ gf_mul(0x0b, state[2][i]) ^ gf_mul(0x0d, state[3][i]);
        temp[2][i] = gf_mul(0x0d, state[0][i]) ^ gf_mul(0x09, state[1][i]) ^ gf_mul(0x0e, state[2][i]) ^ gf_mul(0x0b, state[3][i]);
        temp[3][i] = gf_mul(0x0b, state[0][i]) ^ gf_mul(0x0d, state[1][i]) ^ gf_mul(0x09, state[2][i]) ^ gf_mul(0x0e, state[3][i]);
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] = temp[i][j];
}

void encrypt(uint8_t state[4][4], uint8_t key[4][4], uint8_t last[4][4], uint8_t res[4][4])
{
    uint8_t expand_key[11][4][4];
    key_expand(key, expand_key);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] ^= last[i][j];

    // Round 0
    add_round_key(state, expand_key[0]);

    // Round 1-10
    for (int j = 1; j <= 10; j++)
    {
        sub_bytes(state);
        shift_row(state);
        if (j != 10)
            mix_column(state);
        add_round_key(state, expand_key[j]);
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            res[i][j] = state[j][i];

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            last[i][j] = state[i][j];
}

void encrypt_simd(uint8_t *data, uint8_t *key, __m128i *last, uint8_t *res)
{
    __m128i expand_key[11];
    key_expand_simd(key, expand_key);

    __m128i state = _mm_loadu_si128((const __m128i *)(data));
    state = _mm_xor_si128(state, expand_key[0]);
    state = _mm_xor_si128(state, *last);

    for (int i = 1; i <= 9; i++)
        state = _mm_aesenc_si128(state, expand_key[i]);

    state = _mm_aesenclast_si128(state, expand_key[10]);

    _mm_storeu_si128((__m128i *)(res), state);
    _mm_storeu_si128(last, state);
}

void eq_decrypt_simd(uint8_t *data, uint8_t *key, __m128i *last, uint8_t *res)
{
    __m128i expand_key[11];
    inv_key_expand_simd(key, expand_key);

    __m128i state = _mm_loadu_si128((const __m128i *)(data));
    state = _mm_xor_si128(state, expand_key[10]);

    for (int i = 9; i >= 1; i--)
        state = _mm_aesdec_si128(state, expand_key[i]);

    state = _mm_aesdeclast_si128(state, expand_key[0]);
    state = _mm_xor_si128(state, *last);
    _mm_storeu_si128((__m128i *)(res), state);
}

void decrypt(uint8_t state[4][4], uint8_t key[4][4], uint8_t last[4][4], uint8_t res[4][4])
{
    uint8_t expand_key[11][4][4];
    key_expand(key, expand_key);

    // Round 10~1
    for (int j = 10; j >= 1; j--)
    {
        add_round_key(state, expand_key[j]);
        if (j != 10)
            inv_mix_column(state);
        inv_shift_row(state);
        inv_sub_bytes(state);
    }

    // Round 0
    add_round_key(state, expand_key[0]);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            state[i][j] ^= last[i][j];

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            res[i][j] = state[j][i];
}

int main(void)
{
#ifndef ONLINE_JUDGE
    FILE *in = fopen("aes_cbc_i2.bin", "rb");
    FILE *out = fopen("aeso.bin", "wb");
#else
#define in stdin
#define out stdout
#endif
    // // trivial
    // uint8_t mode;
    // uint8_t key[4][4];
    // uint8_t IV[4][4];
    // uint32_t len;
    // uint8_t res[4][4];

    // fread(&mode, sizeof(uint8_t), 1, in);
    // for (int i = 0; i < 4; i++)
    //     for (int j = 0; j < 4; j++)
    //         fread(&key[j][i], sizeof(uint8_t), 1, in);
    // for (int i = 0; i < 4; i++)
    //     for (int j = 0; j < 4; j++)
    //         fread(&IV[j][i], sizeof(uint8_t), 1, in);
    // fread(&len, sizeof(uint32_t), 1, in);

    // uint8_t last[4][4];
    // for (int i = 0; i < 4; i++)
    //     for (int j = 0; j < 4; j++)
    //         last[i][j] = IV[i][j];

    // init_tables();

    // if (mode == 0x01)
    // {
    //     uint32_t i = 0;
    //     uint8_t data[16];
    //     uint8_t state[4][4];
    //     while (i < len)
    //     {
    //         if (i + 16 > len)
    //         {
    //             fread(data, sizeof(uint8_t), len - i, in);
    //             padding(data, 16 - (len - i));
    //         }
    //         else
    //             fread(data, sizeof(uint8_t), 16, in);

    //         for (int i = 0; i < 4; i++)
    //             for (int j = 0; j < 4; j++)
    //                 state[j][i] = data[i * 4 + j];

    //         encrypt(state, key, last, res);
    //         fwrite(res, sizeof(uint8_t), 16, out);
    //         i += 16;
    //     }

    //     if (i == len)
    //     {
    //         padding(data, 16);

    //         for (int i = 0; i < 4; i++)
    //             for (int j = 0; j < 4; j++)
    //                 state[j][i] = data[i * 4 + j];

    //         encrypt(state, key, last, res);
    //         fwrite(res, sizeof(uint8_t), 16, out);
    //     }
    // }
    // else
    // {
    //     uint32_t i = 0;
    //     uint8_t data[16];
    //     uint8_t state[4][4];

    //     while (i < len)
    //     {
    //         fread(data, sizeof(uint8_t), 16, in);
    //         for (int i = 0; i < 4; i++)
    //             for (int j = 0; j < 4; j++)
    //                 state[j][i] = data[i * 4 + j];

    //         decrypt(state, key, last, res);

    //         int len_ = 16;
    //         if (i + 16 == len)
    //             len_ = 16 - (int)res[3][3];

    //         fwrite(res, sizeof(uint8_t), len_, out);

    //         i += 16;
    //         for (int i = 0; i < 4; i++)
    //             for (int j = 0; j < 4; j++)
    //                 last[j][i] = data[i * 4 + j];
    //     }
    // }

    // SIMD
    uint8_t mode;
    uint8_t key[16];
    uint8_t IV[16];
    uint32_t len;
    uint8_t res[16];
    uint8_t data[16];

    fread(&mode, sizeof(uint8_t), 1, in);
    fread(key, sizeof(uint8_t), 16, in);
    fread(IV, sizeof(uint8_t), 16, in);
    fread(&len, sizeof(uint32_t), 1, in);

    if (mode == 0x01)
    {
        __m128i last = _mm_loadu_si128((const __m128i *)(IV));
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

            encrypt_simd(data, key, &last, res);
            fwrite(res, sizeof(uint8_t), 16, out);
            i += 16;
        }

        if (i == len)
        {
            padding(data, 16);
            encrypt_simd(data, key, &last, res);
            fwrite(res, sizeof(uint8_t), 16, out);
        }
    }
    else
    {
        __m128i last = _mm_loadu_si128((const __m128i *)(IV));
        uint32_t i = 0;

        while (i < len)
        {
            fread(data, sizeof(uint8_t), 16, in);
            eq_decrypt_simd(data, key, &last, res);

            int len_ = 16;
            if (i + 16 == len)
                len_ = 16 - (int)res[15];

            fwrite(res, sizeof(uint8_t), len_, out);

            i += 16;
            last = _mm_loadu_si128((const __m128i *)(data));
        }
    }

    return 0;
}