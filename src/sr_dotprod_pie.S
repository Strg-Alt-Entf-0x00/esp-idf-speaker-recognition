// Copyright 2026 Native Speaker Recognition Project
// ESP32-P4 RV32 PIE INT8 Dot Product Kernel

    .text
    .align  4
    .global sr_dotprod_s8_pie
    .type   sr_dotprod_s8_pie,@function

// esp_err_t sr_dotprod_s8_pie(const int8_t *src1, const int8_t *src2, int32_t *dest, int len);
sr_dotprod_s8_pie:
    // a0: src1 (int8_t*)
    // a1: src2 (int8_t*)
    // a2: dest (int32_t*)
    // a3: len  (int)

    // Verify 16-byte alignment of src1 and src2 pointers
    or      t1, a0, a1
    andi    t1, t1, 15
    bnez    t1, .fallback_s8_ansi

    // Verify length is positive and a multiple of 16
    andi    t1, a3, 15
    bnez    t1, .fallback_s8_ansi
    blez    a3, .fallback_s8_ansi

    srli    t6, a3, 4       // t6 = len / 16

    // Zero 40-bit hardware accumulator
    esp.zero.xacc

    mv      t4, a0          // t4 = src1
    mv      t5, a1          // t5 = src2

    // Preload first 16 bytes of src1 into q0
    esp.vld.128.ip  q0, t4, 16

.loop_s8_body:
    esp.vld.128.ip          q1, t5, 16
    esp.vmulas.s8.xacc.ld.ip q0, t4, 16, q0, q1
    addi    t6, t6, -1
    bgtz    t6, .loop_s8_body

    // Read lower 32 bits of accumulator (shift = 0 via a4)
    li      a4, 0
    esp.srs.s.xacc  t5, a4
    sw      t5, 0(a2)
    li      a0, 0           // ESP_OK
    ret

.fallback_s8_ansi:
    li      t0, 0           // acc = 0
.loop_ansi:
    blez    a3, .done_ansi
    lb      t1, 0(a0)
    lb      t2, 0(a1)
    mul     t1, t1, t2
    add     t0, t0, t1
    addi    a0, a0, 1
    addi    a1, a1, 1
    addi    a3, a3, -1
    j       .loop_ansi
.done_ansi:
    sw      t0, 0(a2)
    li      a0, 0
    ret
