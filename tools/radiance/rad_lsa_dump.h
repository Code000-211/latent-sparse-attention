/* LSA PCA dump (RADIANCE_LSA_DUMP=<dir>): on prefill steps of at least RADIANCE_LSA_DUMP_MIN_T
 * rows (default 64), each attention layer appends to <dir>/r<rank>_L<layer>.bin one record:
 *   int32 magic 0x4C534131, int32 T, int32 kv_heads, int32 free_dims (head_dim - n_rot)
 *   int32 positions[T]
 *   float key [T, kv_heads * free]     -- the normed key heads' position-free dims, side by side
 *   float qsum[T, kv_heads * free]     -- each group's normed query heads summed on those dims
 * Debug only: it syncs the stream once per layer. */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static inline void rad_lsa_dump(RadCtx* c, const RadBatch* batch, rad_buf bq, rad_buf bk,
                                int64_t r0, int64_t T, int64_t n_head, int64_t n_head_kv, int64_t hd,
                                int64_t n_rot, int layer, uint32_t dtype) {
    static const char* dir = std::getenv("RADIANCE_LSA_DUMP");
    if (!dir || !*dir) return;
    static const long min_t = std::getenv("RADIANCE_LSA_DUMP_MIN_T")
                                  ? std::strtol(std::getenv("RADIANCE_LSA_DUMP_MIN_T"), nullptr, 10) : 64;
    if (T < min_t || n_head_kv <= 0 || n_rot >= hd) return;
    if (dtype != RAD_BF16 && dtype != RAD_F32) return;
    void* pq = rad_buf_ptr(c, bq);
    void* pk = rad_buf_ptr(c, bk);
    if (!pq || !pk || !batch->positions) return;
    const int64_t esz = dtype == RAD_F32 ? 4 : 2;
    const int64_t qw = n_head * hd, kw = n_head_kv * hd;
    std::vector<uint8_t> vq((size_t)(T * qw * esz)), vk((size_t)(T * kw * esz));
    std::vector<int32_t> pos((size_t)T);
    const uint8_t* bq0 = (const uint8_t*)pq + r0 * qw * esz;
    const uint8_t* bk0 = (const uint8_t*)pk + r0 * kw * esz;
    RadStream s = rad_stream(c);
    rad_memcpy_async(vq.data(), bq0, T * qw * esz, s);
    rad_memcpy_async(vk.data(), bk0, T * kw * esz, s);
    rad_memcpy_async(pos.data(), batch->positions + r0, T * 4, s);
    rad_stream_sync(s);
    const int64_t fr = hd - n_rot, grp = n_head / n_head_kv, w = n_head_kv * fr;
    std::vector<float> key((size_t)(T * w)), qs((size_t)(T * w), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < n_head_kv; ++h)
            for (int64_t d = 0; d < fr; ++d) {
                key[(size_t)(t * w + h * fr + d)] =
                    rad_load_f32(vk.data(), dtype, t * kw + h * hd + n_rot + d);
                float acc = 0.0f;
                for (int64_t j = 0; j < grp; ++j)
                    acc += rad_load_f32(vq.data(), dtype, t * qw + (h * grp + j) * hd + n_rot + d);
                qs[(size_t)(t * w + h * fr + d)] = acc;
            }
    const std::string path = std::string(dir) + "/r" + std::to_string(rad_rank(c)) + "_L" +
                             std::to_string(layer) + ".bin";
    if (FILE* f = std::fopen(path.c_str(), "ab")) {
        const int32_t hdr[4] = {0x4C534131, (int32_t)T, (int32_t)n_head_kv, (int32_t)fr};
        std::fwrite(hdr, 4, 4, f);
        std::fwrite(pos.data(), 4, (size_t)T, f);
        std::fwrite(key.data(), 4, key.size(), f);
        std::fwrite(qs.data(), 4, qs.size(), f);
        std::fclose(f);
    }
}
