// FFmpeg audio input and output implementation

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/audio_io.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace voxcpm2 {
namespace {

std::string ffmpeg_error(int code) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buf, sizeof(buf));
    return buf;
}

void throw_ffmpeg(int code, const std::string& action) {
    throw std::runtime_error(std::format("{}: {}", action, ffmpeg_error(code)));
}

template <typename T, auto Deleter>
using av_ptr = std::unique_ptr<T, decltype([](T* ptr) { Deleter(&ptr); })>;

using format_input_ptr = av_ptr<AVFormatContext, avformat_close_input>;
using codec_context_ptr = av_ptr<AVCodecContext, avcodec_free_context>;
using frame_ptr = av_ptr<AVFrame, av_frame_free>;
using packet_ptr = av_ptr<AVPacket, av_packet_free>;
using swr_ptr = av_ptr<SwrContext, swr_free>;

bool sample_format_supported(const AVCodec* codec, AVSampleFormat format) {
    if (codec->sample_fmts == nullptr) {
        return true;
    }
    for (const AVSampleFormat* p = codec->sample_fmts; *p != AV_SAMPLE_FMT_NONE; ++p) {
        if (*p == format) {
            return true;
        }
    }
    return false;
}

AVSampleFormat choose_sample_format(const AVCodec* codec) {
    if (sample_format_supported(codec, AV_SAMPLE_FMT_FLT)) {
        return AV_SAMPLE_FMT_FLT;
    }
    if (sample_format_supported(codec, AV_SAMPLE_FMT_FLTP)) {
        return AV_SAMPLE_FMT_FLTP;
    }
    if (sample_format_supported(codec, AV_SAMPLE_FMT_S16)) {
        return AV_SAMPLE_FMT_S16;
    }
    if (sample_format_supported(codec, AV_SAMPLE_FMT_S16P)) {
        return AV_SAMPLE_FMT_S16P;
    }
    if (codec->sample_fmts != nullptr and codec->sample_fmts[0] != AV_SAMPLE_FMT_NONE) {
        return codec->sample_fmts[0];
    }
    return AV_SAMPLE_FMT_FLT;
}

int choose_sample_rate(const AVCodec* codec, int requested) {
    if (codec->supported_samplerates == nullptr) {
        return requested;
    }
    int best = codec->supported_samplerates[0];
    int best_delta = std::abs(best - requested);
    for (const int* p = codec->supported_samplerates; *p != 0; ++p) {
        const int delta = std::abs(*p - requested);
        if (delta < best_delta) {
            best = *p;
            best_delta = delta;
        }
    }
    return best;
}

void encode_and_write(AVFormatContext* format, AVCodecContext* codec, AVFrame* frame) {
    int ret = avcodec_send_frame(codec, frame);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "failed to send audio frame to encoder");
    }

    packet_ptr packet(av_packet_alloc());
    if (not packet) [[unlikely]] {
        throw std::runtime_error("failed to allocate ffmpeg packet");
    }

    while (true) {
        ret = avcodec_receive_packet(codec, packet.get());
        if (ret == AVERROR(EAGAIN) or ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "failed to receive encoded audio packet");
        }
        av_packet_rescale_ts(packet.get(), codec->time_base, format->streams[0]->time_base);
        packet->stream_index = format->streams[0]->index;
        ret = av_interleaved_write_frame(format, packet.get());
        av_packet_unref(packet.get());
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "failed to write encoded audio packet");
        }
    }
}

} // namespace

AudioBuffer read_audio_file(const std::filesystem::path& path, int target_sample_rate) {
    if (target_sample_rate <= 0) [[unlikely]] {
        throw std::runtime_error("target sample rate must be positive");
    }

    AVFormatContext* raw_format = nullptr;
    int ret = avformat_open_input(&raw_format, path.string().c_str(), nullptr, nullptr);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot open audio input " + path.string());
    }
    format_input_ptr format(raw_format);

    ret = avformat_find_stream_info(format.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot read stream info from " + path.string());
    }

    ret = av_find_best_stream(format.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot find audio stream in " + path.string());
    }
    const int stream_index = ret;
    AVStream* stream = format->streams[stream_index];
    const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (decoder == nullptr) [[unlikely]] {
        throw std::runtime_error("cannot find decoder for " + path.string());
    }

    codec_context_ptr codec(avcodec_alloc_context3(decoder));
    if (not codec) [[unlikely]] {
        throw std::runtime_error("failed to allocate ffmpeg decoder context");
    }
    ret = avcodec_parameters_to_context(codec.get(), stream->codecpar);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot copy decoder parameters");
    }
    ret = avcodec_open2(codec.get(), decoder, nullptr);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot open decoder");
    }

    AVChannelLayout output_layout;
    av_channel_layout_default(&output_layout, 1);

    SwrContext* raw_swr = nullptr;
    ret = swr_alloc_set_opts2(
        &raw_swr,
        &output_layout,
        AV_SAMPLE_FMT_FLT,
        target_sample_rate,
        &codec->ch_layout,
        codec->sample_fmt,
        codec->sample_rate,
        0,
        nullptr);
    av_channel_layout_uninit(&output_layout);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot allocate resampler");
    }
    swr_ptr swr(raw_swr);
    ret = swr_init(swr.get());
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot initialize resampler");
    }

    packet_ptr packet(av_packet_alloc());
    frame_ptr frame(av_frame_alloc());
    if (not packet or not frame) [[unlikely]] {
        throw std::runtime_error("failed to allocate ffmpeg decode buffers");
    }

    std::vector<float> samples;
    auto receive_frames = [&] {
        while (true) {
            const int got = avcodec_receive_frame(codec.get(), frame.get());
            if (got == AVERROR(EAGAIN) or got == AVERROR_EOF) {
                break;
            }
            if (got < 0) [[unlikely]] {
                throw_ffmpeg(got, "failed to decode audio frame");
            }
            const int out_count = static_cast<int>(av_rescale_rnd(
                swr_get_delay(swr.get(), codec->sample_rate) + frame->nb_samples,
                target_sample_rate,
                codec->sample_rate,
                AV_ROUND_UP));
            std::vector<float> converted(static_cast<std::size_t>(out_count));
            auto* out_data = reinterpret_cast<std::uint8_t*>(converted.data());
            const int converted_count = swr_convert(
                swr.get(),
                &out_data,
                out_count,
                const_cast<const std::uint8_t**>(frame->extended_data),
                frame->nb_samples);
            if (converted_count < 0) [[unlikely]] {
                throw_ffmpeg(converted_count, "failed to resample decoded audio");
            }
            converted.resize(static_cast<std::size_t>(converted_count));
            samples.insert(samples.end(), converted.begin(), converted.end());
            av_frame_unref(frame.get());
        }
    };

    while (av_read_frame(format.get(), packet.get()) >= 0) {
        if (packet->stream_index == stream_index) {
            ret = avcodec_send_packet(codec.get(), packet.get());
            if (ret < 0) [[unlikely]] {
                throw_ffmpeg(ret, "failed to send packet to decoder");
            }
            receive_frames();
        }
        av_packet_unref(packet.get());
    }
    ret = avcodec_send_packet(codec.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "failed to flush decoder");
    }
    receive_frames();

    if (samples.empty()) [[unlikely]] {
        throw std::runtime_error("decoded audio is empty: " + path.string());
    }
    return AudioBuffer{.sample_rate = target_sample_rate, .channels = 1, .samples = std::move(samples)};
}

void write_audio_file(const std::filesystem::path& path, const AudioBuffer& audio) {
    if (audio.sample_rate <= 0 or audio.channels != 1 or audio.samples.empty()) [[unlikely]] {
        throw std::runtime_error("only non-empty mono float audio can be encoded");
    }

    AVFormatContext* raw_format = nullptr;
    int ret = avformat_alloc_output_context2(&raw_format, nullptr, nullptr, path.string().c_str());
    if (ret < 0 or raw_format == nullptr) [[unlikely]] {
        throw_ffmpeg(ret < 0 ? ret : AVERROR(EINVAL), "cannot infer output format from " + path.string());
    }
    std::unique_ptr<AVFormatContext, decltype([](AVFormatContext* ctx) {
                        if (ctx != nullptr) {
                            if ((ctx->oformat->flags & AVFMT_NOFILE) == 0 and ctx->pb != nullptr) {
                                avio_closep(&ctx->pb);
                            }
                            avformat_free_context(ctx);
                        }
                    })>
        format(raw_format);

    if (format->oformat->audio_codec == AV_CODEC_ID_NONE) [[unlikely]] {
        throw std::runtime_error("output format has no default audio codec: " + path.string());
    }
    const AVCodec* encoder = avcodec_find_encoder(format->oformat->audio_codec);
    if (encoder == nullptr) [[unlikely]] {
        throw std::runtime_error("cannot find encoder for " + path.string());
    }

    AVStream* stream = avformat_new_stream(format.get(), nullptr);
    if (stream == nullptr) [[unlikely]] {
        throw std::runtime_error("failed to create audio stream");
    }

    codec_context_ptr codec(avcodec_alloc_context3(encoder));
    if (not codec) [[unlikely]] {
        throw std::runtime_error("failed to allocate ffmpeg encoder context");
    }
    av_channel_layout_default(&codec->ch_layout, 1);
    codec->sample_fmt = choose_sample_format(encoder);
    codec->sample_rate = choose_sample_rate(encoder, audio.sample_rate);
    codec->time_base = AVRational{1, codec->sample_rate};
    codec->bit_rate = 192000;
    if ((format->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
        codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    ret = avcodec_open2(codec.get(), encoder, nullptr);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot open encoder");
    }
    ret = avcodec_parameters_from_context(stream->codecpar, codec.get());
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot copy encoder parameters");
    }
    stream->time_base = codec->time_base;

    if ((format->oformat->flags & AVFMT_NOFILE) == 0) {
        ret = avio_open(&format->pb, path.string().c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "cannot open audio output " + path.string());
        }
    }

    ret = avformat_write_header(format.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot write output header");
    }

    AVChannelLayout input_layout;
    av_channel_layout_default(&input_layout, 1);
    SwrContext* raw_swr = nullptr;
    ret = swr_alloc_set_opts2(
        &raw_swr,
        &codec->ch_layout,
        codec->sample_fmt,
        codec->sample_rate,
        &input_layout,
        AV_SAMPLE_FMT_FLT,
        audio.sample_rate,
        0,
        nullptr);
    av_channel_layout_uninit(&input_layout);
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot allocate encoder resampler");
    }
    swr_ptr swr(raw_swr);
    ret = swr_init(swr.get());
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot initialize encoder resampler");
    }

    const int input_chunk = codec->frame_size > 0 ? codec->frame_size : 1024;
    std::size_t offset = 0;
    std::int64_t pts = 0;
    while (offset < audio.samples.size()) {
        const int in_count = std::min<int>(input_chunk, static_cast<int>(audio.samples.size() - offset));
        const int out_count = static_cast<int>(av_rescale_rnd(
            swr_get_delay(swr.get(), audio.sample_rate) + in_count,
            codec->sample_rate,
            audio.sample_rate,
            AV_ROUND_UP));

        frame_ptr frame(av_frame_alloc());
        if (not frame) [[unlikely]] {
            throw std::runtime_error("failed to allocate encode frame");
        }
        frame->nb_samples = out_count;
        frame->format = codec->sample_fmt;
        frame->sample_rate = codec->sample_rate;
        ret = av_channel_layout_copy(&frame->ch_layout, &codec->ch_layout);
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "cannot copy frame channel layout");
        }
        ret = av_frame_get_buffer(frame.get(), 0);
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "cannot allocate encode frame buffer");
        }

        const auto* in_data = reinterpret_cast<const std::uint8_t*>(audio.samples.data() + offset);
        const int converted = swr_convert(swr.get(), frame->data, out_count, &in_data, in_count);
        if (converted < 0) [[unlikely]] {
            throw_ffmpeg(converted, "failed to convert audio for encoder");
        }
        frame->nb_samples = converted;
        frame->pts = pts;
        pts += converted;
        if (converted > 0) {
            encode_and_write(format.get(), codec.get(), frame.get());
        }
        offset += static_cast<std::size_t>(in_count);
    }

    while (true) {
        const int out_count = static_cast<int>(av_rescale_rnd(
            swr_get_delay(swr.get(), audio.sample_rate),
            codec->sample_rate,
            audio.sample_rate,
            AV_ROUND_UP));
        if (out_count <= 0) {
            break;
        }
        frame_ptr frame(av_frame_alloc());
        if (not frame) [[unlikely]] {
            throw std::runtime_error("failed to allocate flush frame");
        }
        frame->nb_samples = out_count;
        frame->format = codec->sample_fmt;
        frame->sample_rate = codec->sample_rate;
        ret = av_channel_layout_copy(&frame->ch_layout, &codec->ch_layout);
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "cannot copy flush frame channel layout");
        }
        ret = av_frame_get_buffer(frame.get(), 0);
        if (ret < 0) [[unlikely]] {
            throw_ffmpeg(ret, "cannot allocate flush frame buffer");
        }
        const int converted = swr_convert(swr.get(), frame->data, out_count, nullptr, 0);
        if (converted < 0) [[unlikely]] {
            throw_ffmpeg(converted, "failed to flush encoder resampler");
        }
        if (converted == 0) {
            break;
        }
        frame->nb_samples = converted;
        frame->pts = pts;
        pts += converted;
        encode_and_write(format.get(), codec.get(), frame.get());
    }

    encode_and_write(format.get(), codec.get(), nullptr);
    ret = av_write_trailer(format.get());
    if (ret < 0) [[unlikely]] {
        throw_ffmpeg(ret, "cannot write output trailer");
    }
}

} // namespace voxcpm2
