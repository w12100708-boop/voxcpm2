// Local VoxCPM2 SDPA custom layer for ncnn.
//
// Copyright 2026 Tencent
// Modified for voxcpm2-ncnn.
// SPDX-License-Identifier: BSD-3-Clause

#include "voxcpm2_sdpa.h"

#include "../spirv_cache.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <vector>

#include <cpu.h>
#include <layer_type.h>
#include <modelbin.h>
#include <paramdict.h>

namespace voxcpm2::runtime {
namespace {

std::atomic<bool> g_flash_attention_enabled{true};

#if NCNN_VULKAN
constexpr char kSdpaCrossComp[] = {
#embed "voxcpm2_sdpa_cross.glsl"
    , 0
};
constexpr char kSdpaCrossCmComp[] = {
#embed "voxcpm2_sdpa_cross_cm.glsl"
    , 0
};
constexpr char kSdpaFaComp[] = {
#embed "voxcpm2_sdpa_fa.glsl"
    , 0
};
constexpr char kSdpaFaCmComp[] = {
#embed "voxcpm2_sdpa_fa_cm.glsl"
    , 0
};

int compile_shader(
    SpirvShader shader,
    const char* data,
    int size,
    const ncnn::Option& opt,
    SpirvModule& spirv) {
    return get_cached_spirv(shader, data, size, opt, spirv);
}
#endif

ncnn::Layer* create_voxcpm2_sdpa(void*) {
    return new VoxCPM2SDPA;
}

} // namespace

VoxCPM2SDPA::VoxCPM2SDPA() {
    attn_mask = 0;
    scale = 0.0f;
    kv_cache = 0;
    int8_scale_term = 0;

#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_sdpa_qk_cross = nullptr;
    pipeline_sdpa_qkv_cross = nullptr;
    for (auto*& pipeline : pipeline_sdpa_fa) {
        pipeline = nullptr;
    }
    qk_softmax = nullptr;

    use_flash_attention = false;
    FA_coopmat_M = 0;
    FA_coopmat_N = 0;
    FA_coopmat_K = 0;
    FA_coopmat_subgroup_size = 0;
    FA_UNROLL_SG_M = 1;
    FA_UNROLL_WG_M = 1;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
#endif
}

int VoxCPM2SDPA::load_param(const ncnn::ParamDict& pd) {
    attn_mask = pd.get(5, 0);
    scale = pd.get(6, 0.f);
    kv_cache = pd.get(7, 0);
    int8_scale_term = pd.get(18, 0);

    if (int8_scale_term) {
        support_vulkan = false;
    }

    return 0;
}

int VoxCPM2SDPA::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                         std::vector<ncnn::Mat>& top_blobs,
                         const ncnn::Option& opt) const {
    if (int8_scale_term) {
        return -1;
    }

    const ncnn::Mat& query = bottom_blobs[0];
    const ncnn::Mat& cur_key = bottom_blobs[1];
    const ncnn::Mat& cur_value = bottom_blobs[2];
    const ncnn::Mat& attn_mask_blob = attn_mask ? bottom_blobs[3] : ncnn::Mat();
    const ncnn::Mat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : ncnn::Mat();
    const ncnn::Mat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : ncnn::Mat();

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;
    const float resolved_scale = scale == 0.f ? 1.f / std::sqrt(static_cast<float>(embed_dim)) : scale;
    const int num_heads_per_group = num_heads / num_group;

    ncnn::Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    ncnn::Mat qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
    if (qk_cross.empty()) {
        return -100;
    }

    ncnn::Mat key = cur_key;
    if (past_seqlen > 0) {
        key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (key.empty()) {
            return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; ++q) {
            const ncnn::Mat past_key_head = past_key.channel(q);
            const ncnn::Mat cur_key_head = cur_key.channel(q);
            ncnn::Mat key_head = key.channel(q);

            std::memcpy(key_head.row(0), past_key_head, static_cast<std::size_t>(embed_dim) * past_seqlen * sizeof(float));
            std::memcpy(key_head.row(past_seqlen), cur_key_head, static_cast<std::size_t>(embed_dim) * cur_seqlen * sizeof(float));
        }
    }

    ncnn::Mat value = cur_value;
    if (past_seqlen > 0) {
        value.create(out_embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (value.empty()) {
            return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; ++q) {
            const ncnn::Mat past_value_head = past_value.channel(q);
            const ncnn::Mat cur_value_head = cur_value.channel(q);
            ncnn::Mat value_head = value.channel(q);

            std::memcpy(value_head.row(0), past_value_head, static_cast<std::size_t>(out_embed_dim) * past_seqlen * sizeof(float));
            std::memcpy(value_head.row(past_seqlen), cur_value_head, static_cast<std::size_t>(out_embed_dim) * cur_seqlen * sizeof(float));
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; ++q) {
        const ncnn::Mat query_head = query.channel(q);
        const ncnn::Mat key_head = key.channel(q / num_heads_per_group);
        const ncnn::Mat value_head = value.channel(q / num_heads_per_group);
        ncnn::Mat qk_cross_head = qk_cross.channel(ncnn::get_omp_thread_num());
        ncnn::Mat top_blob_head = top_blob.channel(q);

        for (int i = 0; i < src_seqlen; ++i) {
            const float* qptr = query_head.row(i);
            float* outptr = qk_cross_head.row(i);

            for (int j = 0; j < dst_seqlen; ++j) {
                const float* kptr = key_head.row(j);
                float sum = 0.f;
                for (int k = 0; k < embed_dim; ++k) {
                    sum += qptr[k] * kptr[k];
                }
                outptr[j] = sum * resolved_scale;
            }
        }

        if (attn_mask) {
            const ncnn::Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;

            for (int i = 0; i < src_seqlen; ++i) {
                const float* mptr = maskm.row(i);
                float* outptr = qk_cross_head.row(i);
                for (int j = 0; j < dst_seqlen; ++j) {
                    outptr[j] += mptr[j];
                }
            }
        }

        for (int i = 0; i < src_seqlen; ++i) {
            float* ptr = qk_cross_head.row(i);

            float max = -FLT_MAX;
            for (int j = 0; j < dst_seqlen; ++j) {
                max = std::max(max, ptr[j]);
            }

            float sum = 0.f;
            for (int j = 0; j < dst_seqlen; ++j) {
                ptr[j] = std::exp(ptr[j] - max);
                sum += ptr[j];
            }

            for (int j = 0; j < dst_seqlen; ++j) {
                ptr[j] /= sum;
            }
        }

        for (int i = 0; i < src_seqlen; ++i) {
            const float* qkptr = qk_cross_head.row(i);
            float* outptr = top_blob_head.row(i);

            for (int j = 0; j < out_embed_dim; ++j) {
                float sum = 0.f;
                for (int k = 0; k < dst_seqlen; ++k) {
                    sum += qkptr[k] * value_head.row(k)[j];
                }
                outptr[j] = sum;
            }
        }
    }

    if (kv_cache) {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

int VoxCPM2SDPA::create_pipeline(const ncnn::Option& opt) {
#if NCNN_VULKAN
    if (not opt.use_vulkan_compute or vkdev == nullptr) {
        return 0;
    }

    use_cooperative_matrix = vkdev->info.support_cooperative_matrix() and opt.use_cooperative_matrix and (opt.use_fp16_storage or opt.use_fp16_packed);

    bool use_bf16_cooperative_matrix = false;
    if (vkdev->info.support_bf16_cooperative_matrix() and opt.use_cooperative_matrix and opt.use_bf16_storage) {
        use_cooperative_matrix = true;
        use_bf16_cooperative_matrix = true;
    }

    use_flash_attention = g_flash_attention_enabled.load(std::memory_order_relaxed) and
                          (opt.use_fp16_storage or opt.use_fp16_packed or opt.use_bf16_storage or opt.use_bf16_packed);
    if (use_flash_attention and use_cooperative_matrix) {
        const uint32_t support_subgroup_ops = vkdev->info.support_subgroup_ops();
        const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        use_flash_attention = ((support_subgroup_ops & required_subgroup_ops) == required_subgroup_ops);
    }

    SpirvModule spirv_cross;
    int ret = compile_shader(
        SpirvShader::sdpa_cross,
        kSdpaCrossComp,
        static_cast<int>(sizeof(kSdpaCrossComp) - 1),
        opt,
        spirv_cross);
    if (ret != 0) {
        return ret;
    }

    SpirvModule spirv_cross_cm;
    SpirvModule spirv_fa;
    SpirvModule spirv_fa_cm;
    if (use_cooperative_matrix) {
        ret = compile_shader(
            SpirvShader::sdpa_cross_cooperative_matrix,
            kSdpaCrossCmComp,
            static_cast<int>(sizeof(kSdpaCrossCmComp) - 1),
            opt,
            spirv_cross_cm);
        if (ret != 0) {
            return ret;
        }
    }
    if (use_flash_attention) {
        const char* fa_comp = use_cooperative_matrix ? kSdpaFaCmComp : kSdpaFaComp;
        const int fa_comp_size = use_cooperative_matrix ? static_cast<int>(sizeof(kSdpaFaCmComp) - 1) : static_cast<int>(sizeof(kSdpaFaComp) - 1);
        ret = compile_shader(
            use_cooperative_matrix ? SpirvShader::sdpa_flash_attention_cooperative_matrix
                                   : SpirvShader::sdpa_flash_attention,
            fa_comp,
            fa_comp_size,
            opt,
            use_cooperative_matrix ? spirv_fa_cm : spirv_fa);
        if (ret != 0) {
            return ret;
        }
    }

    if (use_flash_attention) {
        if (use_cooperative_matrix) {
            int M = 1024;
            int N = 1024;
            int K = 1024;

            if (use_bf16_cooperative_matrix) {
                vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, FA_coopmat_M, FA_coopmat_N, FA_coopmat_K, FA_coopmat_subgroup_size);
            } else {
                vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, FA_coopmat_M, FA_coopmat_N, FA_coopmat_K, FA_coopmat_subgroup_size);
            }

            if (FA_coopmat_N != FA_coopmat_K or FA_coopmat_subgroup_size < FA_coopmat_N) {
                use_flash_attention = false;
            } else {
                FA_UNROLL_SG_M = 2;
                FA_UNROLL_WG_M = 2;

                std::vector<ncnn::vk_specialization_type> specializations(1 + 8);
                specializations[0].i = attn_mask;
                specializations[1 + 0].u32 = FA_coopmat_M;
                specializations[1 + 1].u32 = FA_coopmat_N;
                specializations[1 + 2].u32 = FA_coopmat_K;
                specializations[1 + 3].u32 = FA_coopmat_subgroup_size;
                specializations[1 + 4].u32 = FA_UNROLL_SG_M;
                specializations[1 + 5].u32 = FA_UNROLL_WG_M;

                for (int i = 0; i < 8; ++i) {
                    const int max_out_chunks = i + 1;
                    const int unroll_p_n = std::min(4, FA_coopmat_subgroup_size / FA_coopmat_N);
                    specializations[1 + 6].u32 = max_out_chunks;
                    specializations[1 + 7].u32 = unroll_p_n;

                    pipeline_sdpa_fa[i] = new ncnn::Pipeline(vkdev);
                    pipeline_sdpa_fa[i]->set_subgroup_size(FA_coopmat_subgroup_size);
                    pipeline_sdpa_fa[i]->set_local_size_xyz(FA_coopmat_subgroup_size * FA_UNROLL_WG_M, 1, 1);
                    const int create_ret = pipeline_sdpa_fa[i]->create(
                        spirv_fa_cm->data(),
                        spirv_fa_cm->size() * sizeof(std::uint32_t),
                        specializations);
                    if (create_ret != 0) {
                        return create_ret;
                    }
                }
            }
        } else {
            FA_coopmat_M = 4;
            FA_coopmat_N = 32;
            FA_coopmat_K = 32;
            FA_UNROLL_WG_M = 4;
            const int subgroup_size = vkdev->info.subgroup_size();

            std::vector<ncnn::vk_specialization_type> specializations(1 + 6);
            specializations[0].i = attn_mask;
            specializations[1 + 0].u32 = FA_coopmat_M;
            specializations[1 + 1].u32 = FA_coopmat_N;
            specializations[1 + 2].u32 = FA_coopmat_K;
            specializations[1 + 3].u32 = subgroup_size;
            specializations[1 + 4].u32 = FA_UNROLL_WG_M;

            for (int i = 0; i < 8; ++i) {
                const int max_out_chunks = i + 1;
                specializations[1 + 5].u32 = max_out_chunks;

                pipeline_sdpa_fa[i] = new ncnn::Pipeline(vkdev);
                pipeline_sdpa_fa[i]->set_subgroup_size(subgroup_size);
                pipeline_sdpa_fa[i]->set_local_size_xyz(subgroup_size * FA_UNROLL_WG_M, 1, 1);
                const int create_ret = pipeline_sdpa_fa[i]->create(
                    spirv_fa->data(),
                    spirv_fa->size() * sizeof(std::uint32_t),
                    specializations);
                if (create_ret != 0) {
                    return create_ret;
                }
            }
        }
    }

    if (use_cooperative_matrix) {
        int M = 1024;
        int N = 1024;
        int K = 1024;

        if (use_bf16_cooperative_matrix) {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        } else {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        }

        UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
        UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
        UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);
        UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
        UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

        std::vector<ncnn::vk_specialization_type> specializations(13 + 9);
        specializations[0].i = attn_mask;
        specializations[1].f = 0.f;
        specializations[2].i = 0;
        specializations[3].i = 0;
        specializations[4].i = 0;
        specializations[5].i = 0;
        specializations[6].i = 1;
        specializations[7].i = 0;
        specializations[8].i = 0;
        specializations[9].i = 0;
        specializations[10].i = 0;
        specializations[11].i = 0;
        specializations[12].i = 0;
        specializations[13 + 0].u32 = coopmat_M;
        specializations[13 + 1].u32 = coopmat_N;
        specializations[13 + 2].u32 = coopmat_K;
        specializations[13 + 3].u32 = coopmat_subgroup_size;
        specializations[13 + 4].u32 = UNROLL_SG_M;
        specializations[13 + 5].u32 = UNROLL_SG_N;
        specializations[13 + 6].u32 = UNROLL_SG_K;
        specializations[13 + 7].u32 = UNROLL_WG_M;
        specializations[13 + 8].u32 = UNROLL_WG_N;

        pipeline_sdpa_qk_cross = new ncnn::Pipeline(vkdev);
        pipeline_sdpa_qk_cross->set_subgroup_size(coopmat_subgroup_size);
        pipeline_sdpa_qk_cross->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
        int create_ret = pipeline_sdpa_qk_cross->create(
            spirv_cross_cm->data(),
            spirv_cross_cm->size() * sizeof(std::uint32_t),
            specializations);
        if (create_ret != 0) {
            return create_ret;
        }

        specializations[0].i = 0;
        specializations[1].f = 1.f;
        specializations[2].i = 0;
        specializations[3].i = 0;
        specializations[4].i = 0;
        specializations[5].i = 0;
        specializations[6].i = 0;
        specializations[7].i = 0;
        specializations[8].i = 0;
        specializations[9].i = 0;
        specializations[10].i = 0;
        specializations[11].i = 0;
        specializations[12].i = 0;
        specializations[13 + 0].u32 = coopmat_M;
        specializations[13 + 1].u32 = coopmat_N;
        specializations[13 + 2].u32 = coopmat_K;
        specializations[13 + 3].u32 = coopmat_subgroup_size;
        specializations[13 + 4].u32 = UNROLL_SG_M;
        specializations[13 + 5].u32 = UNROLL_SG_N;
        specializations[13 + 6].u32 = UNROLL_SG_K;
        specializations[13 + 7].u32 = UNROLL_WG_M;
        specializations[13 + 8].u32 = UNROLL_WG_N;

        pipeline_sdpa_qkv_cross = new ncnn::Pipeline(vkdev);
        pipeline_sdpa_qkv_cross->set_subgroup_size(coopmat_subgroup_size);
        pipeline_sdpa_qkv_cross->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
        create_ret = pipeline_sdpa_qkv_cross->create(
            spirv_cross_cm->data(),
            spirv_cross_cm->size() * sizeof(std::uint32_t),
            specializations);
        if (create_ret != 0) {
            return create_ret;
        }
    } else {
        {
            std::vector<ncnn::vk_specialization_type> specializations(13);
            specializations[0].i = attn_mask;
            specializations[1].f = 0.f;
            specializations[2].i = 0;
            specializations[3].i = 0;
            specializations[4].i = 0;
            specializations[5].i = 0;
            specializations[6].i = 1;
            specializations[7].i = 0;
            specializations[8].i = 0;
            specializations[9].i = 0;
            specializations[10].i = 0;
            specializations[11].i = 0;
            specializations[12].i = 0;

            pipeline_sdpa_qk_cross = new ncnn::Pipeline(vkdev);
            pipeline_sdpa_qk_cross->set_local_size_xyz(8, 8, 1);
            const int create_ret = pipeline_sdpa_qk_cross->create(
                spirv_cross->data(),
                spirv_cross->size() * sizeof(std::uint32_t),
                specializations);
            if (create_ret != 0) {
                return create_ret;
            }
        }

        {
            std::vector<ncnn::vk_specialization_type> specializations(13);
            specializations[0].i = 0;
            specializations[1].f = 1.f;
            specializations[2].i = 0;
            specializations[3].i = 0;
            specializations[4].i = 0;
            specializations[5].i = 0;
            specializations[6].i = 0;
            specializations[7].i = 0;
            specializations[8].i = 0;
            specializations[9].i = 0;
            specializations[10].i = 0;
            specializations[11].i = 0;
            specializations[12].i = 0;

            pipeline_sdpa_qkv_cross = new ncnn::Pipeline(vkdev);
            pipeline_sdpa_qkv_cross->set_local_size_xyz(8, 8, 1);
            const int create_ret = pipeline_sdpa_qkv_cross->create(
                spirv_cross->data(),
                spirv_cross->size() * sizeof(std::uint32_t),
                specializations);
            if (create_ret != 0) {
                return create_ret;
            }
        }
    }

    qk_softmax = ncnn::create_layer_vulkan(ncnn::LayerType::Softmax);
    qk_softmax->vkdev = vkdev;
    ncnn::ParamDict pd;
    pd.set(0, -1);
    pd.set(1, 1);
    qk_softmax->load_param(pd);
    qk_softmax->load_model(ncnn::ModelBinFromMatArray(nullptr));
    qk_softmax->create_pipeline(opt);
#else
    (void)opt;
#endif

    return 0;
}

int VoxCPM2SDPA::destroy_pipeline(const ncnn::Option& opt) {
#if NCNN_VULKAN
    if (not opt.use_vulkan_compute or vkdev == nullptr) {
        return 0;
    }

    delete pipeline_sdpa_qk_cross;
    pipeline_sdpa_qk_cross = nullptr;

    delete pipeline_sdpa_qkv_cross;
    pipeline_sdpa_qkv_cross = nullptr;

    for (auto*& pipeline : pipeline_sdpa_fa) {
        delete pipeline;
        pipeline = nullptr;
    }

    if (qk_softmax) {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = nullptr;
    }

    use_flash_attention = false;
    FA_coopmat_M = 0;
    FA_coopmat_N = 0;
    FA_coopmat_K = 0;
    FA_coopmat_subgroup_size = 0;
    FA_UNROLL_SG_M = 1;
    FA_UNROLL_WG_M = 1;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
#else
    (void)opt;
#endif

    return 0;
}

#if NCNN_VULKAN
int VoxCPM2SDPA::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                         std::vector<ncnn::VkMat>& top_blobs,
                         ncnn::VkCompute& cmd,
                         const ncnn::Option& opt) const {
    const ncnn::VkMat& query = bottom_blobs[0];
    const ncnn::VkMat& cur_key = bottom_blobs[1];
    const ncnn::VkMat& cur_value = bottom_blobs[2];
    const ncnn::VkMat& attn_mask_blob = attn_mask ? bottom_blobs[3] : ncnn::VkMat();
    const ncnn::VkMat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : ncnn::VkMat();
    const ncnn::VkMat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : ncnn::VkMat();

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;
    const int num_heads_per_group = num_heads / num_group;
    const float resolved_scale = scale == 0.f ? 1.f / std::sqrt(static_cast<float>(embed_dim)) : scale;
    const size_t elemsize = query.elemsize;

    if (past_seqlen > 0) {
        return -1;
    }

    const ncnn::VkMat& key = cur_key;
    const ncnn::VkMat& value = cur_value;

    if (use_flash_attention and embed_dim % 8 == 0 and out_embed_dim % 8 == 0 and out_embed_dim <= FA_coopmat_N * 8) {
        ncnn::VkMat& top_blob = top_blobs[0];
        top_blob.create(out_embed_dim, src_seqlen, num_heads, elemsize, opt.blob_vkallocator);
        if (top_blob.empty()) {
            return -100;
        }

        std::vector<ncnn::VkMat> bindings(5);
        bindings[0] = query;
        bindings[1] = key;
        bindings[2] = value;
        bindings[3] = top_blob;
        bindings[4] = attn_mask_blob;

        std::vector<ncnn::vk_constant_type> constants(13);
        constants[0].f = resolved_scale;
        constants[1].i = src_seqlen;
        constants[2].i = dst_seqlen;
        constants[3].i = embed_dim;
        constants[4].i = out_embed_dim;
        constants[5].i = num_heads;
        constants[6].i = attn_mask_blob.dims and attn_mask_blob.c > 1 ? 3 : attn_mask_blob.dims;
        constants[7].i = num_heads_per_group;
        constants[8].i = query.cstep;
        constants[9].i = key.cstep;
        constants[10].i = value.cstep;
        constants[11].i = top_blob.cstep;
        constants[12].i = attn_mask_blob.cstep;

        ncnn::VkMat dispatcher;
        if (use_cooperative_matrix) {
            const int blocks_x = 1;
            const int blocks_y = (src_seqlen + FA_coopmat_M * FA_UNROLL_SG_M * FA_UNROLL_WG_M - 1) / (FA_coopmat_M * FA_UNROLL_SG_M * FA_UNROLL_WG_M);
            dispatcher.w = (blocks_x * blocks_y) * (FA_coopmat_subgroup_size * FA_UNROLL_WG_M);
        } else {
            const int subgroup_size = vkdev->info.subgroup_size();
            const int blocks_x = 1;
            const int blocks_y = (src_seqlen + FA_coopmat_M - 1) / FA_coopmat_M;
            dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * FA_UNROLL_WG_M);
        }
        dispatcher.h = 1;
        dispatcher.c = num_heads;

        const int max_out_chunks = (out_embed_dim + FA_coopmat_N - 1) / FA_coopmat_N;
        const ncnn::Pipeline* pipeline = pipeline_sdpa_fa[max_out_chunks - 1];
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        return 0;
    }

    ncnn::VkMat qk_cross(dst_seqlen, src_seqlen, num_heads, elemsize, opt.workspace_vkallocator);
    if (qk_cross.empty()) {
        return -100;
    }

    {
        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = query;
        bindings[1] = key;
        bindings[2] = qk_cross;
        bindings[3] = attn_mask_blob;

        std::vector<ncnn::vk_constant_type> constants(11);
        constants[0].f = resolved_scale;
        constants[1].i = src_seqlen;
        constants[2].i = dst_seqlen;
        constants[3].i = embed_dim;
        constants[4].i = num_heads;
        constants[5].i = attn_mask_blob.dims;
        constants[6].i = num_heads_per_group;
        constants[7].i = query.cstep;
        constants[8].i = key.cstep;
        constants[9].i = qk_cross.cstep;
        constants[10].i = attn_mask_blob.cstep;

        if (use_cooperative_matrix) {
            const int blocks_x = (src_seqlen + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (dst_seqlen + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            ncnn::VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = num_heads;

            cmd.record_pipeline(pipeline_sdpa_qk_cross, bindings, constants, dispatcher);
        } else {
            ncnn::VkMat dispatcher;
            dispatcher.w = (dst_seqlen + 3) / 4;
            dispatcher.h = (src_seqlen + 3) / 4;
            dispatcher.c = num_heads;

            cmd.record_pipeline(pipeline_sdpa_qk_cross, bindings, constants, dispatcher);
        }
    }

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    ncnn::VkMat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, elemsize, opt.blob_vkallocator);
    if (top_blob.empty()) {
        return -100;
    }

    {
        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = qk_cross;
        bindings[1] = value;
        bindings[2] = top_blob;
        bindings[3] = ncnn::VkMat();

        std::vector<ncnn::vk_constant_type> constants(11);
        constants[0].f = 1.f;
        constants[1].i = src_seqlen;
        constants[2].i = out_embed_dim;
        constants[3].i = dst_seqlen;
        constants[4].i = num_heads;
        constants[5].i = 0;
        constants[6].i = num_heads_per_group;
        constants[7].i = qk_cross.cstep;
        constants[8].i = value.cstep;
        constants[9].i = top_blob.cstep;
        constants[10].i = 0;

        if (use_cooperative_matrix) {
            const int blocks_x = (src_seqlen + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (out_embed_dim + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            ncnn::VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = num_heads;

            cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
        } else {
            ncnn::VkMat dispatcher;
            dispatcher.w = (out_embed_dim + 3) / 4;
            dispatcher.h = (src_seqlen + 3) / 4;
            dispatcher.c = num_heads;

            cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
        }
    }

    return 0;
}
#endif

int register_voxcpm2_sdpa(ncnn::Net& net) {
    return net.register_custom_layer("VoxCPM2SDPA", create_voxcpm2_sdpa);
}

void set_voxcpm2_sdpa_flash_attention_enabled(bool enabled) {
    g_flash_attention_enabled.store(enabled, std::memory_order_relaxed);
}

} // namespace voxcpm2::runtime
