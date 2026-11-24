idf_component_register(
    SRCS "src/audio_manager.cpp"
    INCLUDE_DIRS "include"
    REQUIRES
        console
        nvs_flash
        espressif__esp_codec_dev
        espressif__esp-dsp
    PRIV_REQUIRES
        driver
        esp_driver_gpio
        esp_driver_i2c
        esp_driver_i2s
        esp_ringbuf
        esp_timer
)
