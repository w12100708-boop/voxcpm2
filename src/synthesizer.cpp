// VoxCPM2 synthesis runtime implementation

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/synthesizer.h"

#include "voxcpm2/paged_kv_cache.h"
#include "voxcpm2/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

namespace voxcpm2 {
namespace {

using json = nlohmann::json;

struct PrefixRow {
    int token_id = 0;
    bool is_audio = false;
    std::vector<float> audio_latent_d_p;
};

ncnn::Mat make_i64_input(int w, int h) {
    ncnn::Mat mat(w, h, std::size_t{8});
    auto* p = static_cast<long long*>(mat.data);
    for (int i = 0; i < w * h; ++i) {
        p[i] = (i % 2 == 0) ? 1 : 0;
    }
    return mat;
}

ncnn::Mat make_i32_input(const std::vector<int>& values) {
    ncnn::Mat mat(static_cast<int>(values.size()));
    std::memcpy(mat.data, values.data(), values.size() * sizeof(int));
    return mat;
}

ncnn::Mat make_f32_input(int w, int h, int c = 1) {
    ncnn::Mat mat = c == 1 ? ncnn::Mat(w, h) : ncnn::Mat(w, h, c);
    auto* p = static_cast<float*>(mat.data);
    for (int i = 0; i < mat.total(); ++i) {
        p[i] = static_cast<float>((i % 17) - 8) / 32.0f;
    }
    return mat;
}

ncnn::Mat make_f32_input1d(int w) {
    ncnn::Mat mat(w);
    auto* p = static_cast<float*>(mat.data);
    for (int i = 0; i < mat.total(); ++i) {
        p[i] = static_cast<float>((i % 17) - 8) / 32.0f;
    }
    return mat;
}

ncnn::Mat make_f32_mat(int w, int h, const std::vector<float>& values) {
    if (static_cast<int>(values.size()) != w * h) [[unlikely]] {
        throw std::runtime_error("make_f32_mat size mismatch");
    }
    ncnn::Mat mat(w, h);
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

ncnn::Mat make_f32_vec(const std::vector<float>& values) {
    ncnn::Mat mat(static_cast<int>(values.size()));
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

ncnn::Mat make_step_mask(int context_len) {
    ncnn::Mat mat(context_len);
    mat.fill(0.0f);
    return mat;
}

std::vector<float> mat_to_vector(const ncnn::Mat& mat) {
    if (mat.elemsize != 4) [[unlikely]] {
        throw std::runtime_error("expected f32 ncnn::Mat");
    }
    std::vector<float> values(mat.total());
    std::memcpy(values.data(), mat.data, values.size() * sizeof(float));
    return values;
}

void print_mat_shape(const std::string& name, const ncnn::Mat& mat) {
    if (mat.empty()) {
        std::println(stderr, "{}: empty", name);
        return;
    }
    std::println(
        stderr,
        "{}: dims={} w={} h={} c={} total={} elemsize={}",
        name,
        mat.dims,
        mat.w,
        mat.h,
        mat.c,
        mat.total(),
        mat.elemsize);
}

ncnn::Mat run_single_input(ncnn::Net& net, const ncnn::Mat& in) {
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract out0");
    }
    return out;
}

ncnn::Mat run_net(ncnn::Net& net, const std::vector<ncnn::Mat>& inputs) {
    ncnn::Extractor ex = net.create_extractor();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::string name = std::format("in{}", i);
        ex.input(name.c_str(), inputs[i]);
    }
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract out0");
    }
    return out;
}

std::vector<float> run_net_vec(ncnn::Net& net, const std::vector<ncnn::Mat>& inputs) {
    return mat_to_vector(run_net(net, inputs));
}

ncnn::Mat run_decoder_step_with_kv(ncnn::Net& net,
                                   const ncnn::Mat& embed,
                                   const ncnn::Mat& mask,
                                   const ncnn::Mat* cos_cache,
                                   const ncnn::Mat* sin_cache,
                                   PagedKvCache& kv_cache,
                                   int attn_count) {
    if (kv_cache.layer_count() != attn_count) [[unlikely]] {
        throw std::runtime_error("paged kv cache layer count mismatch");
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", embed);
    ex.input("in1", mask);
    if (cos_cache != nullptr and sin_cache != nullptr) {
        ex.input("in2", *cos_cache);
        ex.input("in3", *sin_cache);
    }

    std::vector<KvCachePair> materialized;
    materialized.reserve(static_cast<std::size_t>(attn_count));
    for (int i = 0; i < attn_count; ++i) {
        materialized.push_back(kv_cache.materialize(i));
        const std::string k_name = std::format("cache_k{}", i);
        const std::string v_name = std::format("cache_v{}", i);
        ex.input(k_name.c_str(), materialized.back().key);
        ex.input(v_name.c_str(), materialized.back().value);
    }

    for (int i = 0; i < attn_count; ++i) {
        const std::string k_name = std::format("out_cache_k{}", i);
        const std::string v_name = std::format("out_cache_v{}", i);
        ncnn::Mat k_cache;
        ncnn::Mat v_cache;
        if (ex.extract(k_name.c_str(), k_cache) != 0 or k_cache.empty() or
            ex.extract(v_name.c_str(), v_cache) != 0 or v_cache.empty()) [[unlikely]] {
            throw std::runtime_error("failed to extract decoder step kv cache");
        }
        kv_cache.append_from_contiguous(i, k_cache, v_cache);
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract decoder step out0");
    }
    return out;
}

std::vector<float> row_slice(const std::vector<float>& mat, int row, int width) {
    std::vector<float> out(static_cast<std::size_t>(width));
    std::memcpy(out.data(), mat.data() + static_cast<std::size_t>(row) * width, static_cast<std::size_t>(width) * sizeof(float));
    return out;
}

float optimized_scale(const float* positive, const float* negative, int n) {
    double dot = 0.0;
    double norm = 1e-8;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(positive[i]) * negative[i];
        norm += static_cast<double>(negative[i]) * negative[i];
    }
    return static_cast<float>(dot / norm);
}

std::vector<float> cfm_sample(ncnn::Net& dit_estimator,
                              const std::vector<float>& mu,
                              const std::vector<float>& cond,
                              int patch_size,
                              int feat_dim,
                              int timesteps,
                              float cfg_value,
                              std::mt19937& rng) {
    const int latent_size = feat_dim * patch_size;
    std::vector<float> x(static_cast<std::size_t>(latent_size));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : x) {
        value = normal(rng);
    }

    std::vector<float> t_span(static_cast<std::size_t>(timesteps + 1));
    for (int i = 0; i <= timesteps; ++i) {
        const float t = 1.0f - static_cast<float>(i) / static_cast<float>(timesteps);
        t_span[static_cast<std::size_t>(i)] = t + (std::cos(std::numbers::pi_v<float> * 0.5f * t) - 1.0f + t);
    }

    float t = t_span[0];
    float dt = t_span[0] - t_span[1];
    const int zero_init_steps = std::max(1, static_cast<int>((timesteps + 1) * 0.04f));
    if (static_cast<int>(cond.size()) != latent_size) [[unlikely]] {
        throw std::runtime_error("cfm_sample cond size mismatch");
    }

    for (int step = 1; step <= timesteps; ++step) {
        std::vector<float> dphi(static_cast<std::size_t>(latent_size), 0.0f);
        if (step > zero_init_steps) {
            std::vector<float> mu_positive(2048, 0.0f);
            std::memcpy(mu_positive.data(), mu.data(), std::min<std::size_t>(mu.size(), 2048) * sizeof(float));
            std::vector<float> mu_negative(2048, 0.0f);
            const std::vector<float> t_in = {t};
            const std::vector<float> dt_in = {0.0f};

            auto run_dit = [&](const std::vector<float>& hidden_mu) {
                return run_net_vec(
                    dit_estimator,
                    {
                        make_f32_mat(patch_size, feat_dim, x),
                        make_f32_mat(2048, 1, hidden_mu),
                        make_f32_mat(1, 1, t_in),
                        make_f32_mat(patch_size, feat_dim, cond),
                        make_f32_mat(1, 1, dt_in),
                    });
            };

            const std::vector<float> positive = run_dit(mu_positive);
            const std::vector<float> negative = run_dit(mu_negative);
            if (static_cast<int>(positive.size()) != latent_size or static_cast<int>(negative.size()) != latent_size) [[unlikely]] {
                throw std::runtime_error("dit_estimator output size mismatch");
            }
            if (not std::ranges::all_of(positive, [](float v) { return std::isfinite(v); }) or
                not std::ranges::all_of(negative, [](float v) { return std::isfinite(v); })) [[unlikely]] {
                throw std::runtime_error("dit_estimator produced non-finite values");
            }
            const float scale = optimized_scale(positive.data(), negative.data(), latent_size);
            for (int i = 0; i < latent_size; ++i) {
                dphi[static_cast<std::size_t>(i)] =
                    negative[static_cast<std::size_t>(i)] * scale +
                    cfg_value * (positive[static_cast<std::size_t>(i)] - negative[static_cast<std::size_t>(i)] * scale);
            }
        }

        for (int i = 0; i < latent_size; ++i) {
            x[static_cast<std::size_t>(i)] -= dt * dphi[static_cast<std::size_t>(i)];
        }
        t -= dt;
        if (step < timesteps) {
            dt = t - t_span[static_cast<std::size_t>(step + 1)];
        }
    }

    return x;
}

std::vector<float> transpose_d_p_to_p_d(const std::vector<float>& latent, int feat_dim, int patch_size) {
    std::vector<float> out(static_cast<std::size_t>(feat_dim) * patch_size);
    for (int d = 0; d < feat_dim; ++d) {
        for (int p = 0; p < patch_size; ++p) {
            out[static_cast<std::size_t>(p) * feat_dim + d] = latent[static_cast<std::size_t>(d) * patch_size + p];
        }
    }
    return out;
}

std::vector<float> zero_latent_d_p(int feat_dim, int patch_size) {
    return std::vector<float>(static_cast<std::size_t>(feat_dim) * patch_size, 0.0f);
}

std::vector<float> build_vae_latent(const std::vector<std::vector<float>>& patches, int feat_dim, int patch_size) {
    const int total_w = static_cast<int>(patches.size()) * patch_size;
    std::vector<float> out(static_cast<std::size_t>(feat_dim) * total_w, 0.0f);
    for (std::size_t t = 0; t < patches.size(); ++t) {
        const auto& patch = patches[t];
        if (static_cast<int>(patch.size()) != feat_dim * patch_size) [[unlikely]] {
            throw std::runtime_error("latent patch size mismatch");
        }
        for (int d = 0; d < feat_dim; ++d) {
            for (int p = 0; p < patch_size; ++p) {
                out[static_cast<std::size_t>(d) * total_w + t * patch_size + p] =
                    patch[static_cast<std::size_t>(d) * patch_size + p];
            }
        }
    }
    return out;
}

int argmax2(const std::vector<float>& logits) {
    if (logits.size() < 2) {
        return 0;
    }
    return logits[1] > logits[0] ? 1 : 0;
}

int max_patches_for_target(int target_text_token_count, int hard_cap) {
    const int text_tokens = std::max(1, target_text_token_count);
    const int stop_token_cap = static_cast<int>(static_cast<float>(text_tokens) * 6.0f + 10.0f);
    return std::max(1, std::min(hard_cap, stop_token_cap));
}

void pad_audio_to_patch(std::vector<float>& samples, int patch_len, bool left_pad) {
    if (patch_len <= 0) [[unlikely]] {
        throw std::runtime_error("invalid VoxCPM2 audio patch length");
    }
    const int remainder = static_cast<int>(samples.size() % static_cast<std::size_t>(patch_len));
    if (remainder == 0) {
        return;
    }
    const int padding = patch_len - remainder;
    if (left_pad) {
        samples.insert(samples.begin(), static_cast<std::size_t>(padding), 0.0f);
    } else {
        samples.insert(samples.end(), static_cast<std::size_t>(padding), 0.0f);
    }
}

} // namespace

class Synthesizer::Impl {
public:
    explicit Impl(SynthesizerConfig config)
        : model_dir_(std::move(config.model_dir)),
          use_vulkan_(config.use_vulkan),
          threads_(config.threads > 0 ? config.threads : 4),
          vulkan_device_(config.vulkan_device) {
#if NCNN_VULKAN
        if (use_vulkan_) {
            ncnn::create_gpu_instance();
        }
#endif
        load_manifest();
        tokenizer_ = Tokenizer::from_file(model_dir_ / "tokenizer.json");
    }

    ~Impl() {
        nets_.clear();
#if NCNN_VULKAN
        if (use_vulkan_) {
            ncnn::destroy_gpu_instance();
        }
#endif
    }

    AudioBuffer generate(const SynthesisOptions& options) const {
        if (options.text.empty()) [[unlikely]] {
            throw std::runtime_error("text must not be empty");
        }
        const std::vector<std::string> missing = missing_required_components();
        if (not missing.empty()) [[unlikely]] {
            std::string msg = "VoxCPM2 assets are incomplete. Missing ncnn components:";
            for (const auto& name : missing) {
                msg += " " + name;
            }
            throw std::runtime_error(msg);
        }
        if (not options.prompt_text.empty() and not options.prompt_audio.has_value()) [[unlikely]] {
            throw std::runtime_error("prompt text requires prompt audio");
        }
        if (options.inference_timesteps <= 0) [[unlikely]] {
            throw std::runtime_error("inference timesteps must be positive");
        }

        const bool has_prompt = options.prompt_audio.has_value();
        const bool has_reference = options.reference_audio.has_value();

        std::vector<PrefixRow> prefix;
        const std::vector<int> target_ids = tokenizer_.encode(options.text);
        const int target_text_token_count = static_cast<int>(target_ids.size());

        auto append_text_tokens = [&](const std::vector<int>& ids) {
            for (int id : ids) {
                prefix.push_back(PrefixRow{.token_id = id, .is_audio = false, .audio_latent_d_p = {}});
            }
        };
        auto append_audio_patches = [&](const std::vector<std::vector<float>>& patches) {
            for (const auto& patch : patches) {
                prefix.push_back(PrefixRow{.token_id = 0, .is_audio = true, .audio_latent_d_p = patch});
            }
        };
        auto append_reference = [&](const AudioBuffer& audio) {
            prefix.push_back(PrefixRow{.token_id = ref_audio_start_token_, .is_audio = false, .audio_latent_d_p = {}});
            append_audio_patches(encode_audio_to_patches(audio, false));
            prefix.push_back(PrefixRow{.token_id = ref_audio_end_token_, .is_audio = false, .audio_latent_d_p = {}});
        };

        if (has_reference) {
            append_reference(*options.reference_audio);
        }
        if (has_prompt) {
            append_text_tokens(tokenizer_.encode(options.prompt_text + options.text));
            prefix.push_back(PrefixRow{.token_id = audio_start_token_, .is_audio = false, .audio_latent_d_p = {}});
            append_audio_patches(encode_audio_to_patches(*options.prompt_audio, true));
        } else {
            append_text_tokens(target_ids);
            prefix.push_back(PrefixRow{.token_id = audio_start_token_, .is_audio = false, .audio_latent_d_p = {}});
        }

        std::vector<int> token_ids;
        token_ids.reserve(prefix.size());
        for (const auto& row : prefix) {
            token_ids.push_back(row.token_id);
        }

        const int prefix_len = static_cast<int>(prefix.size());
        std::vector<float> text_embed = run_net_vec(net("text_embed"), {make_i32_input(token_ids)});
        if (static_cast<int>(text_embed.size()) != prefix_len * hidden_size_) [[unlikely]] {
            throw std::runtime_error("text_embed output size mismatch");
        }

        std::vector<std::vector<float>> feat_embeds(static_cast<std::size_t>(prefix_len),
                                                    std::vector<float>(hidden_size_, 0.0f));
        for (int i = 0; i < prefix_len; ++i) {
            if (not prefix[static_cast<std::size_t>(i)].is_audio) {
                continue;
            }
            const std::vector<float> feat_input =
                transpose_d_p_to_p_d(prefix[static_cast<std::size_t>(i)].audio_latent_d_p, feat_dim_, patch_size_);
            feat_embeds[static_cast<std::size_t>(i)] =
                run_net_vec(net("feat_encoder"), {make_f32_mat(feat_dim_, patch_size_, feat_input)});
            if (static_cast<int>(feat_embeds[static_cast<std::size_t>(i)].size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("feat_encoder output size mismatch");
            }
        }

        const int max_patches = max_patches_for_target(target_text_token_count, max_generation_patches_);
        std::vector<std::vector<float>> patch_latents;
        patch_latents.reserve(static_cast<std::size_t>(max_patches));
        std::mt19937 rng(0);
        PagedKvCache base_cache(base_attn_count_);
        PagedKvCache residual_cache(residual_attn_count_);
        std::vector<float> zero_embed(hidden_size_, 0.0f);
        std::vector<float> prefix_feat_cond = zero_latent_d_p(feat_dim_, patch_size_);
        std::vector<float> current_lm_hidden;
        std::vector<float> current_residual_hidden;

        for (int pos = 0; pos < prefix_len; ++pos) {
            const PrefixRow& row = prefix[static_cast<std::size_t>(pos)];
            std::vector<float> embed = row.is_audio ? feat_embeds[static_cast<std::size_t>(pos)]
                                                     : row_slice(text_embed, pos, hidden_size_);
            ncnn::Mat cos_cache;
            ncnn::Mat sin_cache;
            make_rope_cache(pos, cos_cache, sin_cache);
            ncnn::Mat base_out = run_decoder_step_with_kv(
                net("base_decoder_step"),
                make_f32_vec(embed),
                make_step_mask(pos + 1),
                &cos_cache,
                &sin_cache,
                base_cache,
                base_attn_count_);
            current_lm_hidden = mat_to_vector(base_out);
            if (static_cast<int>(current_lm_hidden.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("base_decoder_step output size mismatch");
            }
            if (row.is_audio) {
                current_lm_hidden = run_net_vec(net("fsq"), {make_f32_mat(hidden_size_, 1, current_lm_hidden)});
                prefix_feat_cond = row.audio_latent_d_p;
            } else {
                prefix_feat_cond = zero_latent_d_p(feat_dim_, patch_size_);
            }

            std::vector<float> residual_input = run_net_vec(
                net("fusion_proj"),
                {
                    make_f32_mat(hidden_size_, 1, current_lm_hidden),
                    make_f32_mat(hidden_size_, 1, row.is_audio ? feat_embeds[static_cast<std::size_t>(pos)] : zero_embed),
                });
            ncnn::Mat residual_out = run_decoder_step_with_kv(
                net("residual_decoder_step"),
                make_f32_vec(residual_input),
                make_step_mask(pos + 1),
                nullptr,
                nullptr,
                residual_cache,
                residual_attn_count_);
            current_residual_hidden = mat_to_vector(residual_out);
            if (static_cast<int>(current_residual_hidden.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("residual_decoder_step output size mismatch");
            }
        }

        for (int step = 0; step < max_patches; ++step) {
            if (static_cast<int>(current_lm_hidden.size()) != hidden_size_ or
                static_cast<int>(current_residual_hidden.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("decoder step hidden size mismatch");
            }
            if (prefix_len + static_cast<int>(patch_latents.size()) >= rope_original_max_position_embeddings_) [[unlikely]] {
                throw std::runtime_error("VoxCPM2 KV cache reached the exported RoPE range");
            }

            std::vector<float> dit_hidden = run_net_vec(
                net("dit_proj"),
                {
                    make_f32_mat(hidden_size_, 1, current_lm_hidden),
                    make_f32_mat(hidden_size_, 1, current_residual_hidden),
                });
            if (static_cast<int>(dit_hidden.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("dit_proj output size mismatch");
            }

            std::vector<float> latent = cfm_sample(
                net("dit_estimator"),
                dit_hidden,
                prefix_feat_cond,
                patch_size_,
                feat_dim_,
                options.inference_timesteps,
                options.cfg_value,
                rng);
            patch_latents.push_back(latent);
            prefix_feat_cond = latent;

            const std::vector<float> stop_logits =
                run_net_vec(net("stop_head"), {make_f32_mat(hidden_size_, 1, current_lm_hidden)});
            if (step > options.min_patches and argmax2(stop_logits) == 1) {
                break;
            }

            const std::vector<float> feat_input = transpose_d_p_to_p_d(latent, feat_dim_, patch_size_);
            std::vector<float> feat_embed = run_net_vec(net("feat_encoder"), {make_f32_mat(feat_dim_, patch_size_, feat_input)});
            if (static_cast<int>(feat_embed.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("feat_encoder output size mismatch");
            }

            const int patch_pos = prefix_len + static_cast<int>(patch_latents.size()) - 1;
            ncnn::Mat cos_cache;
            ncnn::Mat sin_cache;
            make_rope_cache(patch_pos, cos_cache, sin_cache);
            ncnn::Mat base_out = run_decoder_step_with_kv(
                net("base_decoder_step"),
                make_f32_vec(feat_embed),
                make_step_mask(patch_pos + 1),
                &cos_cache,
                &sin_cache,
                base_cache,
                base_attn_count_);
            current_lm_hidden = mat_to_vector(base_out);
            if (static_cast<int>(current_lm_hidden.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("base_decoder_step output size mismatch");
            }

            std::vector<float> processed = run_net_vec(net("fsq"), {make_f32_mat(hidden_size_, 1, current_lm_hidden)});
            current_lm_hidden = processed;
            std::vector<float> residual_input = run_net_vec(
                net("fusion_proj"),
                {
                    make_f32_mat(hidden_size_, 1, processed),
                    make_f32_mat(hidden_size_, 1, feat_embed),
                });
            ncnn::Mat residual_out = run_decoder_step_with_kv(
                net("residual_decoder_step"),
                make_f32_vec(residual_input),
                make_step_mask(patch_pos + 1),
                nullptr,
                nullptr,
                residual_cache,
                residual_attn_count_);
            current_residual_hidden = mat_to_vector(residual_out);
            if (static_cast<int>(current_residual_hidden.size()) != hidden_size_) [[unlikely]] {
                throw std::runtime_error("residual_decoder_step output size mismatch");
            }
        }

        std::vector<float> vae_latent = build_vae_latent(patch_latents, latent_dim_, patch_size_);
        std::vector<float> samples = run_net_vec(
            net("audio_vae_decoder"),
            {make_f32_mat(static_cast<int>(patch_latents.size()) * patch_size_, latent_dim_, vae_latent)});
        return AudioBuffer{.sample_rate = out_sample_rate_, .channels = 1, .samples = std::move(samples)};
    }

    void smoke_components() const {
        const std::vector<std::string> missing = missing_required_components();
        if (not missing.empty()) [[unlikely]] {
            std::string msg = "cannot smoke-test incomplete VoxCPM2 assets. Missing:";
            for (const auto& name : missing) {
                msg += " " + name;
            }
            throw std::runtime_error(msg);
        }

        print_mat_shape("text_embed", run_single_input(net("text_embed"), make_i64_input(4, 1)));
        {
            PagedKvCache cache(base_attn_count_);
            ncnn::Mat cos_cache;
            ncnn::Mat sin_cache;
            make_rope_cache(0, cos_cache, sin_cache);
            print_mat_shape(
                "base_decoder_step",
                run_decoder_step_with_kv(
                    net("base_decoder_step"),
                    make_f32_input1d(hidden_size_),
                    make_step_mask(1),
                    &cos_cache,
                    &sin_cache,
                    cache,
                    base_attn_count_));
        }
        {
            PagedKvCache cache(residual_attn_count_);
            print_mat_shape(
                "residual_decoder_step",
                run_decoder_step_with_kv(
                    net("residual_decoder_step"),
                    make_f32_input1d(hidden_size_),
                    make_step_mask(1),
                    nullptr,
                    nullptr,
                    cache,
                    residual_attn_count_));
        }
        print_mat_shape("feat_encoder", run_single_input(net("feat_encoder"), make_f32_input(feat_dim_, patch_size_)));
        print_mat_shape("fsq", run_single_input(net("fsq"), make_f32_input(hidden_size_, 1)));
        print_mat_shape(
            "fusion_proj",
            run_net(net("fusion_proj"), {make_f32_input(hidden_size_, 1), make_f32_input(hidden_size_, 1)}));
        print_mat_shape(
            "dit_proj",
            run_net(net("dit_proj"), {make_f32_input(hidden_size_, 1), make_f32_input(hidden_size_, 1)}));
        print_mat_shape(
            "dit_estimator",
            run_net(
                net("dit_estimator"),
                {
                    make_f32_input(patch_size_, feat_dim_),
                    make_f32_input(hidden_size_, 1),
                    make_f32_input(1, 1),
                    make_f32_input(patch_size_, feat_dim_),
                    make_f32_input(1, 1),
                }));
        print_mat_shape("stop_head", run_single_input(net("stop_head"), make_f32_input(hidden_size_, 1)));
        print_mat_shape("audio_vae_encoder", run_single_input(net("audio_vae_encoder"), make_f32_input(patch_size_ * chunk_size_ * 2, 1)));
        print_mat_shape("audio_vae_decoder", run_single_input(net("audio_vae_decoder"), make_f32_input(patch_size_ * 2, latent_dim_)));
    }

    int input_sample_rate() const {
        return encode_sample_rate_;
    }

    int output_sample_rate() const {
        return out_sample_rate_;
    }

    std::vector<std::string> missing_required_components() const {
        static const std::vector<std::string> required = {
            "text_embed",
            "base_decoder_step",
            "residual_decoder_step",
            "fsq",
            "fusion_proj",
            "dit_proj",
            "stop_head",
            "feat_encoder",
            "dit_estimator",
            "audio_vae_encoder",
            "audio_vae_decoder",
        };
        std::vector<std::string> missing;
        for (const auto& name : required) {
            if (not has_net(name)) {
                missing.push_back(name);
            }
        }
        return missing;
    }

private:
    void load_manifest() {
        const std::filesystem::path manifest_path = model_dir_ / "model.json";
        std::ifstream ifs(manifest_path);
        if (not ifs) [[unlikely]] {
            throw std::runtime_error("cannot open " + manifest_path.string());
        }
        ifs >> manifest_;

        const std::string model_type = manifest_.value("model_type", "");
        if (model_type != "voxcpm2_tts") [[unlikely]] {
            throw std::runtime_error("model.json model_type must be voxcpm2_tts");
        }

        const auto setting = manifest_.value("setting", json::object());
        patch_size_ = setting.value("patch_size", patch_size_);
        feat_dim_ = setting.value("feat_dim", feat_dim_);
        latent_dim_ = setting.value("latent_dim", latent_dim_);
        chunk_size_ = setting.value("chunk_size", chunk_size_);
        encode_sample_rate_ = setting.value("encode_sample_rate", encode_sample_rate_);
        out_sample_rate_ = setting.value("out_sample_rate", out_sample_rate_);
        base_attn_count_ = setting.value("base_attn_cnt", base_attn_count_);
        residual_attn_count_ = setting.value("residual_attn_cnt", residual_attn_count_);

        const auto rope = setting.value("rope", json::object());
        rope_head_dim_ = rope.value("rope_head_dim", rope_head_dim_);
        rope_theta_ = rope.value("rope_theta", rope_theta_);
        rope_original_max_position_embeddings_ =
            rope.value("original_max_position_embeddings", rope_original_max_position_embeddings_);
        rope_short_factor_.clear();
        rope_long_factor_.clear();
        if (rope.contains("short_factor") and rope["short_factor"].is_array()) {
            for (const auto& item : rope["short_factor"]) {
                rope_short_factor_.push_back(item.get<float>());
            }
        }
        if (rope.contains("long_factor") and rope["long_factor"].is_array()) {
            for (const auto& item : rope["long_factor"]) {
                rope_long_factor_.push_back(item.get<float>());
            }
        }

        const auto tokens = setting.value("tokens", json::object());
        audio_start_token_ = tokens.value("audio_start", audio_start_token_);
        ref_audio_start_token_ = tokens.value("ref_audio_start", ref_audio_start_token_);
        ref_audio_end_token_ = tokens.value("ref_audio_end", ref_audio_end_token_);

        const auto params = manifest_.value("params", json::object());
        static const std::unordered_set<std::string> supported = {
            "text_embed",
            "base_decoder_step",
            "residual_decoder_step",
            "fsq",
            "fusion_proj",
            "dit_proj",
            "stop_head",
            "feat_encoder",
            "dit_estimator",
            "audio_vae_encoder",
            "audio_vae_decoder",
        };
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (supported.contains(it.key())) {
                load_net(it.key(), it.value());
            }
        }
    }

    void load_net(const std::string& name, const json& params) {
        if (not params.is_object()) {
            return;
        }
        const std::string param_file = params.value("param", "");
        const std::string bin_file = params.value("bin", "");
        if (param_file.empty() or bin_file.empty()) {
            return;
        }

        auto net_ptr = std::make_shared<ncnn::Net>();
        net_ptr->opt = option_for_net(name);
#if NCNN_VULKAN
        if (use_vulkan_ and vulkan_device_ >= 0) {
            net_ptr->set_vulkan_device(vulkan_device_);
        }
#endif
        if (net_ptr->load_param((model_dir_ / param_file).string().c_str()) != 0) [[unlikely]] {
            throw std::runtime_error("failed to load param for " + name);
        }
        if (net_ptr->load_model((model_dir_ / bin_file).string().c_str()) != 0) [[unlikely]] {
            throw std::runtime_error("failed to load bin for " + name);
        }
        nets_[name] = std::move(net_ptr);
    }

    ncnn::Net& net(const std::string& name) const {
        auto it = nets_.find(name);
        if (it == nets_.end()) [[unlikely]] {
            throw std::runtime_error("missing net: " + name);
        }
        return *it->second;
    }

    bool has_net(const std::string& name) const {
        return nets_.contains(name);
    }

    std::vector<std::vector<float>> encode_audio_to_patches(const AudioBuffer& audio, bool left_pad) const {
        if (audio.channels != 1 or audio.sample_rate != encode_sample_rate_) [[unlikely]] {
            throw std::runtime_error(std::format(
                "reference/prompt audio must be mono {} Hz before encoding",
                encode_sample_rate_));
        }
        if (audio.samples.empty()) [[unlikely]] {
            throw std::runtime_error("reference/prompt audio is empty");
        }

        std::vector<float> samples = audio.samples;
        pad_audio_to_patch(samples, patch_size_ * chunk_size_, left_pad);

        const std::vector<float> encoded =
            run_net_vec(net("audio_vae_encoder"), {make_f32_mat(static_cast<int>(samples.size()), 1, samples)});
        const int patch_values = latent_dim_ * patch_size_;
        if (patch_values <= 0 or encoded.empty() or static_cast<int>(encoded.size()) % patch_values != 0) [[unlikely]] {
            throw std::runtime_error("audio_vae_encoder output size mismatch");
        }

        const int patch_count = static_cast<int>(encoded.size()) / patch_values;
        const int total_patch_width = patch_count * patch_size_;
        std::vector<std::vector<float>> patches;
        patches.reserve(static_cast<std::size_t>(patch_count));
        for (int t = 0; t < patch_count; ++t) {
            std::vector<float> patch(static_cast<std::size_t>(patch_values));
            for (int d = 0; d < latent_dim_; ++d) {
                for (int p = 0; p < patch_size_; ++p) {
                    patch[static_cast<std::size_t>(d) * patch_size_ + p] =
                        encoded[static_cast<std::size_t>(d) * total_patch_width + static_cast<std::size_t>(t) * patch_size_ + p];
                }
            }
            patches.push_back(std::move(patch));
        }
        return patches;
    }

    void make_rope_cache(int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const {
        if (rope_head_dim_ <= 0 or rope_head_dim_ % 2 != 0) [[unlikely]] {
            throw std::runtime_error("invalid VoxCPM2 rope head dim");
        }
        const int half_dim = rope_head_dim_ / 2;
        const std::vector<float>& factor =
            (position_id + 1 > rope_original_max_position_embeddings_ and not rope_long_factor_.empty())
                ? rope_long_factor_
                : rope_short_factor_;
        if (static_cast<int>(factor.size()) < half_dim) [[unlikely]] {
            throw std::runtime_error("VoxCPM2 rope factor table is missing or too short");
        }

        cos_cache.create(rope_head_dim_);
        sin_cache.create(rope_head_dim_);
        auto* cos_ptr = static_cast<float*>(cos_cache.data);
        auto* sin_ptr = static_cast<float*>(sin_cache.data);
        for (int j = 0; j < half_dim; ++j) {
            const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(rope_head_dim_);
            const float inv_freq = 1.0f / std::pow(rope_theta_, exponent);
            const float freq = (static_cast<float>(position_id) / factor[static_cast<std::size_t>(j)]) * inv_freq;
            const float c = std::cos(freq);
            const float s = std::sin(freq);
            cos_ptr[j] = c;
            cos_ptr[j + half_dim] = c;
            sin_ptr[j] = s;
            sin_ptr[j + half_dim] = s;
        }
    }

    ncnn::Option option_for_net(const std::string& name) const {
        ncnn::Option opt;
        opt.num_threads = threads_;
        opt.use_bf16_storage = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_arithmetic = false;
        opt.use_int8_storage = false;
        opt.use_int8_packed = false;
        opt.use_int8_arithmetic = false;
        opt.use_vulkan_compute = use_vulkan_;
        if (name == "text_embed") {
            opt.use_vulkan_compute = false;
        }
        if (name == "dit_estimator") {
            opt.use_packing_layout = false;
        }
        return opt;
    }

    std::filesystem::path model_dir_;
    json manifest_;
    Tokenizer tokenizer_;
    std::unordered_map<std::string, std::shared_ptr<ncnn::Net>> nets_;
    bool use_vulkan_ = false;
    int threads_ = 4;
    int vulkan_device_ = 0;

    int hidden_size_ = 2048;
    int patch_size_ = 4;
    int feat_dim_ = 64;
    int latent_dim_ = 64;
    int chunk_size_ = 640;
    int audio_start_token_ = 101;
    int ref_audio_start_token_ = 103;
    int ref_audio_end_token_ = 104;
    int encode_sample_rate_ = 16000;
    int out_sample_rate_ = 48000;
    int base_attn_count_ = 28;
    int residual_attn_count_ = 8;
    int rope_head_dim_ = 128;
    int rope_original_max_position_embeddings_ = 32768;
    int max_generation_patches_ = 2000;
    float rope_theta_ = 10000.0f;
    std::vector<float> rope_short_factor_;
    std::vector<float> rope_long_factor_;
};

Synthesizer::Synthesizer(SynthesizerConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Synthesizer::~Synthesizer() = default;
Synthesizer::Synthesizer(Synthesizer&&) noexcept = default;
Synthesizer& Synthesizer::operator=(Synthesizer&&) noexcept = default;

AudioBuffer Synthesizer::generate(const SynthesisOptions& options) const {
    return impl_->generate(options);
}

void Synthesizer::smoke_components() const {
    impl_->smoke_components();
}

int Synthesizer::input_sample_rate() const {
    return impl_->input_sample_rate();
}

int Synthesizer::output_sample_rate() const {
    return impl_->output_sample_rate();
}

std::vector<std::string> Synthesizer::missing_required_components() const {
    return impl_->missing_required_components();
}

} // namespace voxcpm2
