add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.debug", "mode.release")

set_project("voxcpm2-ncnn")
set_version("0.1.0")
set_languages("c++23", "c11")

option("profile")
    set_default(false)
    set_showmenu(true)
    set_description("Enable coarse VoxCPM2 synthesis profile timings")
    add_defines("VOXCPM2_ENABLE_PROFILE")
option_end()


add_requires("ncnn master", { configs = { vulkan = true } })
add_requires("nlohmann_json")
add_requires("cli11")
add_requires("indicators")
add_requires("ffmpeg", {
    system = true,
    configs = {
        avcodec = true,
        avformat = true,
        avutil = true,
        swresample = true,
    },
})

target("voxcpm2_ncnn")
    set_kind("static")
    add_options("profile")
    add_includedirs("include", { public = true })
    add_files("src/audio_io.cpp", "src/kvcache.cpp", "src/ncnn_layers/dtype_adapter/voxcpm2_dtype_adapter.cpp", "src/ncnn_layers/sdpa/voxcpm2_sdpa.cpp", "src/ncnn_layers/timestep_embedding/voxcpm2_timestep_embedding.cpp", "src/progress.cpp", "src/synthesizer.cpp", "src/tokenizer.cpp")
    if has_config("profile") then
        add_files("src/profile.cpp")
    end
    add_packages("ncnn", "nlohmann_json", "ffmpeg")

target("voxcpm2")
    set_kind("binary")
    add_options("profile")
    add_files("src/main.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("cli11", "indicators", "ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")

target("test_progress")
    set_kind("binary")
    set_group("test")
    add_includedirs("src")
    add_files("tests/test_progress.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_tokenizer")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_tokenizer.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_api_contract")
    set_kind("binary")
    set_group("test")
    add_options("profile")
    add_files("tests/test_api_contract.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")
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
