add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.debug", "mode.release")

set_project("voxcpm2-ncnn")
set_version("0.1.0")
set_languages("c++23", "c11")


add_requires("ncnn master", { configs = { vulkan = true } })
add_requires("nlohmann_json")
add_requires("cli11")
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
    add_includedirs("include", { public = true })
    add_files("src/audio_io.cpp", "src/paged_kv_cache.cpp", "src/synthesizer.cpp", "src/tokenizer.cpp")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")

target("voxcpm2")
    set_kind("binary")
    add_files("src/main.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("cli11", "ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")

target("test_tokenizer")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_tokenizer.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_paged_kv_cache")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_paged_kv_cache.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")

target("test_api_contract")
    set_kind("binary")
    set_group("test")
    add_files("tests/test_api_contract.cpp")
    add_deps("voxcpm2_ncnn")
    add_packages("ncnn", "nlohmann_json", "ffmpeg")
    set_rundir("$(projectdir)")
    add_tests("default")
