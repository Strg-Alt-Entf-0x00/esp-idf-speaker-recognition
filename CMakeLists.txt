idf_component_register(
    SRCS
        "src/sr_audio_frontend.c"
        "src/sr_model.c"
        "src/sr_simd_pie.c"
        "src/sr_parallel.c"
        "src/sr_dotprod_pie.S"
        "src/sr_engine.c"
        "src/sr_database.c"
    INCLUDE_DIRS
        "include"
    PRIV_INCLUDE_DIRS
        "src"
    REQUIRES
        esp-dsp
        esp_timer
        nvs_flash
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -fno-math-errno -funroll-loops)
