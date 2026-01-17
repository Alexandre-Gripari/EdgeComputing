/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <persondet_post_processing.h>
#include <stdio.h>
#include "fsl_debug_console.h"
#include "image.h"
#include "static_image_data.h"
#include "image_utils.h"
#include "model.h"
#include "output_postproc.h"
#include "timer.h"

extern "C"
{

#define MODEL_IN_W 160
#define MODEL_IN_H 128
#define MODEL_IN_C 3
#define MODEL_IN_COLOR_BGR 1

#define BOX_SCORE_THRESHOLD 0.50
#define MAX_OD_BOX_CNT 10

#define BENCHMARK_ITERATIONS 100
#define WARMUP_ITERATIONS 10
#define IDLE_TIME_US 5000000

    typedef struct tagODResult_t
    {
        union
        {
            int16_t xyxy[4];
            struct
            {
                int16_t x1;
                int16_t y1;
                int16_t x2;
                int16_t y2;
            };
        };
        float score;
        int label;
    } ODResult_t;

    ODResult_t s_odRets[MAX_OD_BOX_CNT];
    int s_odRetCnt = 0;
    uint32_t s_infUs = 0;
    uint32_t s_preUs = 0;  
    uint32_t s_postUs = 0; 

    void person_det()
    {
        tensor_dims_t inputDims;
        tensor_type_t inputType;
        uint8_t *inputData;

        tensor_dims_t outputDims;
        tensor_type_t outputType;
        uint8_t *outputData;
        size_t arenaSize;

        if (MODEL_Init() != kStatus_Success)
        {
            PRINTF("Failed initializing model\r\n");
            for (;;)
            {
            }
        }

        size_t usedSize = MODEL_GetArenaUsedBytes(&arenaSize);
        PRINTF("Arena: %d/%d kB (%0.2f%%) used\r\n", usedSize / 1024, arenaSize / 1024, 100.0 * usedSize / arenaSize);

        inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
        outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);

        TfLiteTensor *outputTensor = MODEL_GetOutputTensor(0);

        fast_detection::PostProcessParams postProcessParams = fast_detection::PostProcessParams{
            .inputImgRows = inputDims.data[1],
            .inputImgCols = inputDims.data[2],
            .originalImageWidth = 320,
            .originalImageHeight = 240,
            .threshold = 0.50,
            .nms = 0.45,
            .topN = 10};

        std::vector<fast_detection::DetectionResult> results;
        DetectorPostProcess postProcess = DetectorPostProcess((const TfLiteTensor *)outputTensor,
                                                              results, postProcessParams);

        PRINTF("- PHASE IDLE (ATTENTE %d ms) -\r\n", IDLE_TIME_US / 1000);

        uint64_t idle_start = TIMER_GetTimeInUS();
        while ((TIMER_GetTimeInUS() - idle_start) < IDLE_TIME_US)
        {
            __asm volatile("nop");
        }
        PRINTF("- FIN IDLE -\r\n");

        PRINTF("- DEBUT DU BATCH TEST -\r\n");

        for (int loop_idx = 0; loop_idx < BENCHMARK_ITERATIONS; loop_idx++)
        {

            for (int img_idx = 0; img_idx < DATASET_IMG_COUNT; img_idx++)
            {
                const uint8_t *pCurrentImg = dataset_images[img_idx];
                uint64_t timestamp_pre_start = TIMER_GetTimeInUS();
                for (int i = 0; i < MODEL_IN_W * MODEL_IN_H * MODEL_IN_C; i++)
                {
                    inputData[i] = pCurrentImg[i] ^ 0x80;
                }
                uint64_t timestamp_pre_end = TIMER_GetTimeInUS();
                s_preUs = (uint32_t)(timestamp_pre_end - timestamp_pre_start);
                results.clear();

                uint64_t timestamp_inf_start = TIMER_GetTimeInUS();
                MODEL_RunInference();
                uint64_t timestamp_inf_end = TIMER_GetTimeInUS();
                s_infUs = (uint32_t)(timestamp_inf_end - timestamp_inf_start);

                s_odRetCnt = 0;
                uint64_t timestamp_post_start = TIMER_GetTimeInUS();
                if (!postProcess.DoPostProcess())
                {
                    PRINTF("Post-processing failed ID:%d\r\n", img_idx);
                }

                uint64_t timestamp_post_end = TIMER_GetTimeInUS();
                s_postUs = (uint32_t)(timestamp_post_end - timestamp_post_start);

                if (loop_idx >= WARMUP_ITERATIONS)
                {
                    uint64_t current_ts = TIMER_GetTimeInUS();
                    
                    PRINTF("LOG_DATA: TS=%u PRE=%u INF=%u POST=%u\r\n",
                           (uint32_t)current_ts, 
                           s_preUs,
                           s_infUs,
                           s_postUs);
                }
            }
        }
        PRINTF("- FIN DE L'EXECUTION DU BATCH TEST -\r\n");

        PRINTF("- PHASE IDLE (ATTENTE %d ms) -\r\n", IDLE_TIME_US / 1000);

        idle_start = TIMER_GetTimeInUS();
        while ((TIMER_GetTimeInUS() - idle_start) < IDLE_TIME_US)
        {
            __asm volatile("nop");
        }
        PRINTF("- FIN IDLE -\r\n");

        PRINTF("- FIN DU BENCHMARK -\r\n");
    }
}
