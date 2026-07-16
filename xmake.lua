add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.debug", "mode.release")

set_project("voxcpm2-ncnn")
set_version("0.1.0")
set_languages("c++23", "c11")

option("profile")
    set_default(false)
    set_showmenu(true)
    set_description("Enable VoxCPM2 synthesis profiling instrumentation")
    add_defines("VOXCPM2_ENABLE_PROFILE")
option_end()

add_requires("ncnn master", { configs = { vulkan = true, simpleomp = is_plat("android") } })
add_requires("nlohmann_json")
if not is_plat("android") then
    add_requires("cli11")
    add_requires("indicators")
    add_requires("crow v1.3.2", { configs = { ssl = false, zlib = false } })
    add_requires("ffmpeg", {
        system = true,
        configs = {
            avcodec = true,
            avformat = true,
            avutil = true,
            swresample = true,
        },
    })
end

target("voxcpm2_ncnn")
    set_kind("static")
    add_options("profile")
    add_includedirs("include", { public = true })
    add_files("src/kvcache.cpp", "src/model_manifest.cpp", "src/ncnn_layers/dtype_adapter/voxcpm2_dtype_adapter.cpp", "src/ncnn_layers/sdpa/voxcpm2_sdpa.cpp", "src/ncnn_layers/timestep_embedding/voxcpm2_timestep_embedding.cpp", "src/progress.cpp", "src/synthesizer.cpp", "src/tokenizer.cpp")
    if has_config("profile") then
        add_files("src/profile.cpp")
    end
    add_packages("ncnn", { public = true })
    add_packages("nlohmann_json")

target("voxcpm2_ncnn_shared")
    set_kind("shared")
    set_basename("voxcpm2_ncnn")
    set_targetdir("$(builddir)/$(plat)/$(arch)/$(mode)/shared")
    add_rules("utils.symbols.export_all", { export_classes = true })
    add_options("profile")
    add_includedirs("include", { public = true })
    add_files("src/kvcache.cpp", "src/model_manifest.cpp", "src/ncnn_layers/dtype_adapter/voxcpm2_dtype_adapter.cpp", "src/ncnn_layers/sdpa/voxcpm2_sdpa.cpp", "src/ncnn_layers/timestep_embedding/voxcpm2_timestep_embedding.cpp", "src/progress.cpp", "src/synthesizer.cpp", "src/tokenizer.cpp")
    if has_config("profile") then
        add_files("src/profile.cpp")
    end
    add_packages("ncnn", { public = true })
    add_packages("nlohmann_json")

target("voxcpm2_jni")
    set_kind("shared")
    add_includedirs("include")
    add_files("src/jni.cpp")
    add_deps("voxcpm2_ncnn_shared")

target("voxcpm2_audio_ffmpeg")
    set_kind("static")
    add_includedirs("include", { public = true })
    add_files("src/audio_io.cpp")
    add_packages("ffmpeg", { public = true })

target("voxcpm2")
    set_kind("binary")
    add_options("profile")
    add_files("src/main.cpp")
    add_deps("voxcpm2_ncnn", "voxcpm2_audio_ffmpeg")
    add_packages("cli11", "indicators")
    set_rundir("$(projectdir)")

target("voxcpm2-server")
    set_kind("binary")
    add_options("profile")
    add_includedirs("src")
    add_files("src/server.cpp", "src/server_api.cpp")
    add_deps("voxcpm2_ncnn", "voxcpm2_audio_ffmpeg")
    add_packages("cli11", "crow", "nlohmann_json")
    set_rundir("$(projectdir)")

target("test_progress")
    set_kind("binary")
    set_group("test")
    add_includedirs("src")
    add_files("tests/test_progress.cpp")
    add_deps("voxcpm2_ncnn")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_tokenizer")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_tokenizer.cpp")
    add_deps("voxcpm2_ncnn")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_model_manifest")
    set_kind("binary")
    set_group("test")
    add_includedirs("src")
    add_files("tests/test_model_manifest.cpp", "src/model_manifest.cpp")
    add_packages("nlohmann_json")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_api_contract")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_api_contract.cpp")
    add_deps("voxcpm2_ncnn", "voxcpm2_audio_ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_audio_io")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_audio_io.cpp")
    add_deps("voxcpm2_audio_ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_server_api")
    set_kind("binary")
    set_group("test")
    add_includedirs("include", "src")
    add_files("tests/test_server_api.cpp", "src/server_api.cpp")
    add_packages("nlohmann_json")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_shared_library")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_shared_library.cpp")
    add_deps("voxcpm2_ncnn_shared")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_kvcache")
    set_kind("binary")
    set_group("test")
    add_options("profile")
    add_includedirs("src")
    add_files("tests/test_kvcache.cpp")
    add_deps("voxcpm2_ncnn")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_voxcpm2_sdpa")
    set_kind("binary")
    set_group("test")
    add_includedirs("src")
    add_files("tests/test_voxcpm2_sdpa.cpp", "src/ncnn_layers/sdpa/voxcpm2_sdpa.cpp")
    add_packages("ncnn")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_voxcpm2_timestep_embedding")
    set_kind("binary")
    set_group("test")
    add_includedirs("src")
    add_files("tests/test_voxcpm2_timestep_embedding.cpp", "src/ncnn_layers/timestep_embedding/voxcpm2_timestep_embedding.cpp")
    add_packages("ncnn")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_voxcpm2_dtype_adapter")
    set_kind("binary")
    set_group("test")
    add_includedirs("src")
    add_files("tests/test_voxcpm2_dtype_adapter.cpp", "src/ncnn_layers/dtype_adapter/voxcpm2_dtype_adapter.cpp")
    add_packages("ncnn")
    set_rundir("$(projectdir)")
    add_tests("default")

if is_plat("android") then
    for _, name in ipairs({
        "voxcpm2_audio_ffmpeg",
        "voxcpm2",
        "voxcpm2-server",
        "test_progress",
        "test_tokenizer",
        "test_model_manifest",
        "test_api_contract",
        "test_audio_io",
        "test_server_api",
        "test_shared_library",
        "test_kvcache",
        "test_voxcpm2_sdpa",
        "test_voxcpm2_timestep_embedding",
        "test_voxcpm2_dtype_adapter",
    }) do
        target(name)
            set_default(false)
            before_build(function (target)
                raise("target(%s) is not supported on Android", target:name())
            end)
    end
else
    target("voxcpm2_jni")
        set_default(false)
        before_build(function (target)
            raise("target(%s) is only supported on Android", target:name())
        end)
end
