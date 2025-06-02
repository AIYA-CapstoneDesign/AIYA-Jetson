/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "yolov8.h"
#include "objectTracker.h"
#include "tensorConvert.h"

#include "cudaDraw.h"
#include "cudaFont.h"
#include "cudaMappedMemory.h"

#include "commandLine.h"
#include "filesystem.h"
#include "json.hpp"
#include "logging.h"
#include <algorithm>
#include <vector>

#define OUTPUT_NUM 0

#define PERSON_DETECTION

#define CHECK_NULL_STR(x) (x != NULL) ? x : "NULL"
//#define DEBUG_CLUSTERING

// constructor
yolov8::yolov8(float meanPixel) : tensorNet() {
  mTracker = NULL;
  mMeanPixel = meanPixel;
  mLineWidth = 2.0f;

  mNumClasses = 0;
  mClassColors = NULL;

  mDetectionSets = NULL;
  mDetectionSet = 0;
  mMaxDetections = 0;
  mOverlayAlpha = YOLOV8_DEFAULT_ALPHA;

  mConfidenceThreshold = YOLOV8_DEFAULT_CONFIDENCE_THRESHOLD;
  mNMSThreshold = YOLOV8_DEFAULT_NMS_THRESHOLD;
}

// destructor
yolov8::~yolov8() {
  SAFE_DELETE(mTracker);

  CUDA_FREE_HOST(mDetectionSets);
  CUDA_FREE_HOST(mClassColors);
}

// Create (UFF)
yolov8 *yolov8::Create(const char *model, float threshold, const char *input,
                       const Dims3 &inputDims, const char *output,
                       uint32_t maxBatchSize, precisionType precision,
                       deviceType device, bool allowGPUFallback) {
  yolov8 *net = new yolov8();

  if (!net)
    return NULL;

  LogInfo("\n");
  LogInfo("yolov8 -- loading detection network model from:\n");
  LogInfo("          -- model        %s\n", CHECK_NULL_STR(model));
  LogInfo("          -- input_blob   '%s'\n", CHECK_NULL_STR(input));
  LogInfo("          -- output_blob  '%s'\n", CHECK_NULL_STR(output));
  LogInfo("          -- threshold    %f\n", threshold);
  LogInfo("          -- batch_size   %u\n\n", maxBatchSize);

  // create list of output names
  std::vector<std::string> output_blobs;

  if (output != NULL)
    output_blobs.push_back(output);

  // load the model
  if (!net->LoadNetwork(model, NULL, input, inputDims, output_blobs,
                        maxBatchSize, precision, device, allowGPUFallback)) {
    LogError(LOG_TRT "yolov8 -- failed to initialize.\n");
    return NULL;
  }

  // allocate detection sets
  if (!net->allocDetections())
    return NULL;

  // load class descriptions
  net->loadClassInfo(NULL);
  net->loadClassColors(NULL);

  // set the specified threshold
  net->SetConfidenceThreshold(threshold);
  net->SetNMSThreshold(YOLOV8_DEFAULT_NMS_THRESHOLD);

  LoadCOCOClasses(net->mClassDesc);

  return net;
}

// allocDetections
bool yolov8::allocDetections() {
  // YOLOv8 output format: [batch, 84, 8400] -> after shiftDims: [84, 8400]
  // Where 84 = 4 (bbox coords) + 80 (class scores)
  // And 8400 = number of detection anchors

  // For YOLOv8, the output dimensions after shiftDims are:
  // d[0] = 84 (4 bbox coords + num_classes)
  // d[1] = 8400 (max detections)
  mNumClasses = DIMS_C(mOutputs[OUTPUT_NUM].dims) - 4; // 84 - 4 = 80
  LogInfo(LOG_TRT "yolov8 -- numClasses: %u\n", mNumClasses);
  mMaxDetections = DIMS_H(mOutputs[OUTPUT_NUM].dims); // 8400

  LogVerbose(LOG_TRT "yolov8 -- maximum bounding boxes:   %u\n",
             mMaxDetections);

  // allocate array to store detection results
  const size_t det_size =
      sizeof(Detection) * mNumDetectionSets * mMaxDetections;

  if (!cudaAllocMapped((void **)&mDetectionSets, det_size))
    return false;

  memset(mDetectionSets, 0, det_size);
  return true;
}

// loadClassInfo
bool yolov8::loadClassInfo(const char *filename) {
  // If filename is NULL or loading from file fails, use hardcoded COCO 80
  // classes
  if (!filename ||
      !LoadClassLabels(filename, mClassDesc, mClassSynset, mNumClasses)) {
    LogInfo(LOG_TRT "loading hardcoded COCO 80 class names\n");

    // Load hardcoded COCO 80 classes
    LoadCOCOClasses(mClassDesc);

    // Generate synsets for COCO classes
    mClassSynset.clear();
    mClassSynset.reserve(80);
    for (int i = 0; i < 80; i++) {
      char synset[16];
      sprintf(synset, "coco_%03d", i);
      mClassSynset.push_back(synset);
    }

    mNumClasses = 80;
  }

  if (IsModelType(MODEL_UFF))
    mNumClasses = mClassDesc.size();

  LogInfo(LOG_TRT "yolov8 -- number of object classes:  %u\n", mNumClasses);

  if (filename != NULL)
    mClassPath = locateFile(filename);
  else
    mClassPath = "hardcoded_coco_80";

  return true;
}

// loadClassColors
bool yolov8::loadClassColors(const char *filename) {
  return LoadClassColors(filename, &mClassColors, mNumClasses,
                         YOLOV8_DEFAULT_ALPHA);
}

// Detect
int yolov8::Detect(float *input, uint32_t width, uint32_t height,
                   Detection **detections, uint32_t overlay) {
  return Detect((void *)input, width, height, IMAGE_RGBA32F, detections,
                overlay);
}

// Detect
int yolov8::Detect(void *input, uint32_t width, uint32_t height,
                   imageFormat format, Detection **detections,
                   uint32_t overlay) {
  Detection *det = mDetectionSets + mDetectionSet * GetMaxDetections();

  if (detections != NULL)
    *detections = det;

  mDetectionSet++;

  if (mDetectionSet >= mNumDetectionSets)
    mDetectionSet = 0;

  return Detect(input, width, height, format, det, overlay);
}

// Detect
int yolov8::Detect(float *input, uint32_t width, uint32_t height,
                   Detection *detections, uint32_t overlay) {
  return Detect((void *)input, width, height, IMAGE_RGBA32F, detections,
                overlay);
}

// Detect
int yolov8::Detect(void *input, uint32_t width, uint32_t height,
                   imageFormat format, Detection *detections,
                   uint32_t overlay) {
  // verify parameters
  if (!input || width == 0 || height == 0 || !detections) {
    LogError(LOG_TRT "yolov8::Detect( 0x%p, %u, %u ) -> invalid parameters\n",
             input, width, height);
    return -1;
  }

  if (!imageFormatIsRGB(format)) {
    LogError(LOG_TRT "yolov8::Detect() -- unsupported image format (%s)\n",
             imageFormatToStr(format));
    LogError(LOG_TRT "                       supported formats are:\n");
    LogError(LOG_TRT "                          * rgb8\n");
    LogError(LOG_TRT "                          * rgba8\n");
    LogError(LOG_TRT "                          * rgb32f\n");
    LogError(LOG_TRT "                          * rgba32f\n");

    return false;
  }

  // apply input pre-processing
  if (!preProcess(input, width, height, format))
    return -1;

  // process model with TensorRT
  PROFILER_BEGIN(PROFILER_NETWORK);

  if (!ProcessNetwork())
    return -1;

  PROFILER_END(PROFILER_NETWORK);

  // post-processing / clustering
  const int numDetections =
      postProcess(input, width, height, format, detections);

  // render the overlay
  if (overlay != 0 && numDetections > 0) {
    if (!Overlay(input, input, width, height, format, detections, numDetections,
                 overlay))
      LogError(LOG_TRT "yolov8::Detect() -- failed to render overlay\n");
  }

  // wait for GPU to complete work
  // CUDA(cudaDeviceSynchronize());	// BUG is this needed here?

  // return the number of detections
  return numDetections;
}

// preProcess
bool yolov8::preProcess(void *input, uint32_t width, uint32_t height,
                        imageFormat format) {
  PROFILER_BEGIN(PROFILER_PREPROCESS);

  // YOLOv8 전처리: Center padding with keep aspect ratio
  uint32_t model_width = GetInputWidth();   // 640
  uint32_t model_height = GetInputHeight(); // 640

  // Calculate scaling ratio (keeping aspect ratio)
  float scale = fmin((float)model_width / (float)width,
                     (float)model_height / (float)height);

  // Calculate new dimensions after scaling
  uint32_t new_width = (uint32_t)((float)width * scale);
  uint32_t new_height = (uint32_t)((float)height * scale);

  // Calculate center padding
  int pad_x = (model_width - new_width) / 2;
  int pad_y = (model_height - new_height) / 2;

  LogVerbose(LOG_TRT "YOLOv8 preprocess: input %dx%d -> scaled %dx%d -> padded "
                     "%dx%d (pad: %d,%d)\n",
             width, height, new_width, new_height, model_width, model_height,
             pad_x, pad_y);

  if (IsModelType(MODEL_ONNX) || IsModelType(MODEL_ENGINE)) {
    // YOLOv8 전처리: RGB, 0-1 정규화, 패딩 색상 (114,114,114)/255
    if (CUDA_FAILED(cudaTensorNormPaddingRGB(
            input, format, width, height, mInputs[0].CUDA, model_width,
            model_height, make_float2(0.0f, 1.0f), // 정규화 범위 [0, 1]
            pad_x, pad_y, pad_x, pad_y,
            make_float3(114.0f / 255.0f, 114.0f / 255.0f,
                        114.0f / 255.0f), // YOLOv8 패딩 색상
            GetStream()))) {
      LogError(LOG_TRT
               "yolov8::Detect() -- cudaTensorNormPaddingRGB() failed\n");
      return false;
    }
  } else {
    // Unsupported
    LogError(LOG_TRT "yolov8::Detect() -- unsupported model type\n");
    return false;
  }

  PROFILER_END(PROFILER_PREPROCESS);
  return true;
}

// postProcess
int yolov8::postProcess(void *input, uint32_t width, uint32_t height,
                        imageFormat format, Detection *detections) {
  PROFILER_BEGIN(PROFILER_POSTPROCESS);

  // parse the bounding boxes
  int numDetections = 0;

  numDetections = postProcessYOLOv8(detections, width, height);

  numDetections = applyNMS(detections, numDetections, mNMSThreshold);

  // verify the bounding boxes are within the bounds of the image
  for (int n = 0; n < numDetections; n++) {
    if (detections[n].Top < 0)
      detections[n].Top = 0;

    if (detections[n].Left < 0)
      detections[n].Left = 0;

    if (detections[n].Right >= width)
      detections[n].Right = width - 1;

    if (detections[n].Bottom >= height)
      detections[n].Bottom = height - 1;
  }

  // update tracking
  if (mTracker != NULL && mTracker->IsEnabled())
    numDetections = mTracker->Process(input, width, height, format, detections,
                                      numDetections);

  PROFILER_END(PROFILER_POSTPROCESS);
  return numDetections;
}

// postProcessSSD_ONNX
int yolov8::postProcessYOLOv8(Detection *detections, uint32_t width,
                              uint32_t height) {
  int numDetections = 0;

  float *rawDetections = mOutputs[OUTPUT_NUM].CPU;

  // YOLOv8 output format: [84, 8400] where data is stored as:
  // [x1, x2, ..., x8400, y1, y2, ..., y8400, w1, w2, ..., w8400, h1, h2, ...,
  // h8400,
  //  class0_conf1, class0_conf2, ..., class0_conf8400, class1_conf1, ...]
  const uint32_t numBoxesClasses = DIMS_C(mOutputs[OUTPUT_NUM].dims);  // 84
  const uint32_t numMaxDetections = DIMS_H(mOutputs[OUTPUT_NUM].dims); // 8400

  // Calculate ratio for original image and paddings (전처리와 동일한 방식)
  uint32_t model_width = GetInputWidth();   // 640
  uint32_t model_height = GetInputHeight(); // 640

  // Calculate scaling ratio (keeping aspect ratio) - 전처리와 동일
  float scale = fmin((float)model_width / (float)width,
                     (float)model_height / (float)height);

  // Calculate new dimensions after scaling
  uint32_t new_width = (uint32_t)((float)width * scale);
  uint32_t new_height = (uint32_t)((float)height * scale);

  // Calculate center padding - 전처리와 동일
  int pad_x = (model_width - new_width) / 2;
  int pad_y = (model_height - new_height) / 2;

  for (uint16_t n = 0; n < numMaxDetections; n++) {
    uint8_t maxClass = -1;
    float maxScore = -1000.0f;

    // TensorRT output은 (84, 8400) 형태로 저장됨
    // transpose 방식으로 접근: [x_n, y_n, w_n, h_n, class0_n, class1_n, ...]
    float cx =
        rawDetections[0 * numMaxDetections + n]; // x values start at index 0
    float cy =
        rawDetections[1 * numMaxDetections + n]; // y values start at index 8400
    float w = rawDetections[2 * numMaxDetections +
                            n]; // w values start at index 16800
    float h = rawDetections[3 * numMaxDetections +
                            n]; // h values start at index 25200

    // Find the class with maximum confidence (transpose 방식)
    for (uint8_t m = 0; m < mNumClasses; m++) {
      float score = rawDetections[(4 + m) * numMaxDetections +
                                  n]; // class scores start at index 4*8400

      if (score < mConfidenceThreshold)
        continue;

      if (n < 10)
        LogInfo(LOG_TRT "yolov8 -- score: %f\n", score);

      if (score > maxScore) {
        maxScore = score;
        maxClass = m;
      }
    }
	
    if (maxScore < mConfidenceThreshold)
      continue;

#ifdef PERSON_DETECTION
    if (maxClass != 0)
      continue;
#endif

    // Convert from center format to corner format and apply scaling
    // YOLOv8 출력은 640x640 모델 좌표계에서 나옴
    // 1. padding 제거 2. 스케일링 역변환으로 원본 이미지 좌표로 변환
    float left = (cx - w / 2 - pad_x) / scale;
    float top = (cy - h / 2 - pad_y) / scale;
    float right = (cx + w / 2 - pad_x) / scale;
    float bottom = (cy + h / 2 - pad_y) / scale;

    detections[numDetections].TrackID = -1;
    detections[numDetections].ClassID = maxClass;
    detections[numDetections].Confidence = maxScore;
    detections[numDetections].Left = left;
    detections[numDetections].Top = top;
    detections[numDetections].Right = right;
    detections[numDetections].Bottom = bottom;

    numDetections++;
  }

  return numDetections;
}

// sortDetections
void yolov8::sortDetections(Detection *detections, int numDetections) {
  if (numDetections < 2)
    return;

  // Use std::sort with lambda comparator for O(n log n) performance
  // Sort in descending order by confidence score
  std::sort(detections, detections + numDetections,
            [](const Detection &a, const Detection &b) {
              return a.Confidence > b.Confidence;
            });
}

// applyClassWiseNMS - Apply class-wise hard Non-Maximum Suppression
int yolov8::applyNMS(Detection *detections, int numDetections,
                     float nmsThreshold) {
  if (numDetections <= 1 || nmsThreshold <= 0.0f)
    return numDetections;

  // First, sort detections by confidence (already done in sortDetections)
  // But we need to ensure it's done for NMS to work properly
  sortDetections(detections, numDetections);

  // Create a vector to track which detections to keep
  std::vector<bool> keep(numDetections, true);
  int numKept = 0;

  // Apply NMS for each class separately
  for (int i = 0; i < numDetections; i++) {
    if (!keep[i])
      continue;

    uint32_t currentClass = detections[i].ClassID;

    // Count this detection as kept
    numKept++;

    // Compare with all subsequent detections of the same class
    for (int j = i + 1; j < numDetections; j++) {
      if (!keep[j] || detections[j].ClassID != currentClass)
        continue;

      // Calculate IoU between current detection and candidate
      float iou = detections[i].IOU(detections[j]);

      // If IoU exceeds threshold, suppress the lower confidence detection
      if (iou > nmsThreshold) {
        keep[j] = false;
      }
    }
  }

  // Compact the array by moving kept detections to the front
  int writeIndex = 0;
  for (int i = 0; i < numDetections; i++) {
    if (keep[i]) {
      if (writeIndex != i) {
        detections[writeIndex] = detections[i];
      }
      writeIndex++;
    }
  }

  return writeIndex; // Return the new number of detections
}

// from detectNet.cu
cudaError_t cudaDetectionOverlay(void *input, void *output, uint32_t width,
                                 uint32_t height, imageFormat format,
                                 yolov8::Detection *detections,
                                 int numDetections, float4 *colors);

// Overlay
bool yolov8::Overlay(void *input, void *output, uint32_t width, uint32_t height,
                     imageFormat format, Detection *detections,
                     uint32_t numDetections, uint32_t flags) {
  PROFILER_BEGIN(PROFILER_VISUALIZE);

  if (flags == 0) {
    LogError(LOG_TRT "yolov8 -- Overlay() was called with OVERLAY_NONE, "
                     "returning false\n");
    return false;
  }

  // if input and output are different images, copy the input to the output
  // first then overlay the bounding boxes, ect. on top of the output image
  if (input != output) {
    if (CUDA_FAILED(cudaMemcpy(output, input,
                               imageFormatSize(format, width, height),
                               cudaMemcpyDeviceToDevice))) {
      LogError(LOG_TRT "yolov8 -- Overlay() failed to copy input image to "
                       "output image\n");
      return false;
    }
  }

  // make sure there are actually detections
  if (numDetections <= 0) {
    PROFILER_END(PROFILER_VISUALIZE);
    return true;
  }

  // bounding box overlay
  if (flags & OVERLAY_BOX) {
    if (CUDA_FAILED(cudaDetectionOverlay(input, output, width, height, format,
                                         detections, numDetections,
                                         mClassColors)))
      return false;
  }

  // bounding box lines
  if (flags & OVERLAY_LINES) {
    for (uint32_t n = 0; n < numDetections; n++) {
      const Detection *d = detections + n;
      const float4 &color = mClassColors[d->ClassID];

      CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Top,
                        d->Right, d->Top, color, mLineWidth));
      CUDA(cudaDrawLine(input, output, width, height, format, d->Right, d->Top,
                        d->Right, d->Bottom, color, mLineWidth));
      CUDA(cudaDrawLine(input, output, width, height, format, d->Left,
                        d->Bottom, d->Right, d->Bottom, color, mLineWidth));
      CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Top,
                        d->Left, d->Bottom, color, mLineWidth));
    }
  }

  // class label overlay
  if ((flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) ||
      (flags & OVERLAY_TRACKING)) {
    static cudaFont *font = NULL;

    // make sure the font object is created
    if (!font) {
      font = cudaFont::Create(adaptFontSize(width)); // 20.0f

      if (!font) {
        LogError(LOG_TRT "yolov8 -- Overlay() was called with OVERLAY_FONT, "
                         "but failed to create cudaFont()\n");
        return false;
      }
    }

    // draw each object's description
#ifdef BATCH_TEXT
    std::vector<std::pair<std::string, int2>> labels;
#endif
    for (uint32_t n = 0; n < numDetections; n++) {
      const char *className = GetClassDesc(detections[n].ClassID);
      const float confidence = detections[n].Confidence * 100.0f;
      const int2 position =
          make_int2(detections[n].Left + 5, detections[n].Top + 3);

      char buffer[256];
      char *str = buffer;

      if (flags & OVERLAY_LABEL)
        str += sprintf(str, "%s ", className);

      if (flags & OVERLAY_TRACKING && detections[n].TrackID >= 0)
        str += sprintf(str, "%i ", detections[n].TrackID);

      if (flags & OVERLAY_CONFIDENCE)
        str += sprintf(str, "%.1f%%", confidence);

#ifdef BATCH_TEXT
      labels.push_back(std::pair<std::string, int2>(buffer, position));
#else
      float4 color = make_float4(255, 255, 255, 255);

      if (detections[n].TrackID >= 0)
        color.w *= 1.0f - (fminf(detections[n].TrackLost, 15.0f) / 15.0f);

      font->OverlayText(output, format, width, height, buffer, position.x,
                        position.y, color);
#endif
    }

#ifdef BATCH_TEXT
    font->OverlayText(output, format, width, height, labels,
                      make_float4(255, 255, 255, 255));
#endif
  }

  PROFILER_END(PROFILER_VISUALIZE);
  return true;
}

// OverlayFlagsFromStr
uint32_t yolov8::OverlayFlagsFromStr(const char *str_user) {
  if (!str_user)
    return OVERLAY_DEFAULT;

  // copy the input string into a temporary array,
  // because strok modifies the string
  const size_t str_length = strlen(str_user);
  const size_t max_length = 256;

  if (str_length == 0)
    return OVERLAY_DEFAULT;

  if (str_length >= max_length) {
    LogError(LOG_TRT "yolov8::OverlayFlagsFromStr() overlay string exceeded "
                     "max length of %zu characters ('%s')",
             max_length, str_user);
    return OVERLAY_DEFAULT;
  }

  char str[max_length];
  strcpy(str, str_user);

  // tokenize string by delimiters ',' and '|'
  const char *delimiters = ",|";
  char *token = strtok(str, delimiters);

  if (!token)
    return OVERLAY_DEFAULT;

  // look for the tokens:  "box", "label", "default", and "none"
  uint32_t flags = OVERLAY_NONE;

  while (token != NULL) {
    if (strcasecmp(token, "box") == 0)
      flags |= OVERLAY_BOX;
    else if (strcasecmp(token, "label") == 0 ||
             strcasecmp(token, "labels") == 0)
      flags |= OVERLAY_LABEL;
    else if (strcasecmp(token, "conf") == 0 ||
             strcasecmp(token, "confidence") == 0)
      flags |= OVERLAY_CONFIDENCE;
    else if (strcasecmp(token, "track") == 0 ||
             strcasecmp(token, "tracking") == 0)
      flags |= OVERLAY_TRACKING;
    else if (strcasecmp(token, "line") == 0 || strcasecmp(token, "lines") == 0)
      flags |= OVERLAY_LINES;
    else if (strcasecmp(token, "default") == 0)
      flags |= OVERLAY_DEFAULT;

    token = strtok(NULL, delimiters);
  }

  return flags;
}

// SetOverlayAlpha
void yolov8::SetOverlayAlpha(float alpha) {
  const uint32_t numClasses = GetNumClasses();

  for (uint32_t n = 0; n < numClasses; n++)
    mClassColors[n].w = alpha;

  mOverlayAlpha = alpha;
}

// LoadCOCO80Classes - Load hardcoded COCO 80 class names
void yolov8::LoadCOCOClasses(std::vector<std::string> &classNames) {
  classNames.clear();
  classNames.reserve(80);

  // COCO 80 classes in the correct order
  classNames.push_back("person");
  classNames.push_back("bicycle");
  classNames.push_back("car");
  classNames.push_back("motorcycle");
  classNames.push_back("airplane");
  classNames.push_back("bus");
  classNames.push_back("train");
  classNames.push_back("truck");
  classNames.push_back("boat");
  classNames.push_back("traffic light");
  classNames.push_back("fire hydrant");
  classNames.push_back("stop sign");
  classNames.push_back("parking meter");
  classNames.push_back("bench");
  classNames.push_back("bird");
  classNames.push_back("cat");
  classNames.push_back("dog");
  classNames.push_back("horse");
  classNames.push_back("sheep");
  classNames.push_back("cow");
  classNames.push_back("elephant");
  classNames.push_back("bear");
  classNames.push_back("zebra");
  classNames.push_back("giraffe");
  classNames.push_back("backpack");
  classNames.push_back("umbrella");
  classNames.push_back("handbag");
  classNames.push_back("tie");
  classNames.push_back("suitcase");
  classNames.push_back("frisbee");
  classNames.push_back("skis");
  classNames.push_back("snowboard");
  classNames.push_back("sports ball");
  classNames.push_back("kite");
  classNames.push_back("baseball bat");
  classNames.push_back("baseball glove");
  classNames.push_back("skateboard");
  classNames.push_back("surfboard");
  classNames.push_back("tennis racket");
  classNames.push_back("bottle");
  classNames.push_back("wine glass");
  classNames.push_back("cup");
  classNames.push_back("fork");
  classNames.push_back("knife");
  classNames.push_back("spoon");
  classNames.push_back("bowl");
  classNames.push_back("banana");
  classNames.push_back("apple");
  classNames.push_back("sandwich");
  classNames.push_back("orange");
  classNames.push_back("broccoli");
  classNames.push_back("carrot");
  classNames.push_back("hot dog");
  classNames.push_back("pizza");
  classNames.push_back("donut");
  classNames.push_back("cake");
  classNames.push_back("chair");
  classNames.push_back("couch");
  classNames.push_back("potted plant");
  classNames.push_back("bed");
  classNames.push_back("dining table");
  classNames.push_back("toilet");
  classNames.push_back("tv");
  classNames.push_back("laptop");
  classNames.push_back("mouse");
  classNames.push_back("remote");
  classNames.push_back("keyboard");
  classNames.push_back("cell phone");
  classNames.push_back("microwave");
  classNames.push_back("oven");
  classNames.push_back("toaster");
  classNames.push_back("sink");
  classNames.push_back("refrigerator");
  classNames.push_back("book");
  classNames.push_back("clock");
  classNames.push_back("vase");
  classNames.push_back("scissors");
  classNames.push_back("teddy bear");
  classNames.push_back("hair drier");
  classNames.push_back("toothbrush");

  LogInfo(LOG_TRT "loaded %zu COCO class names\n", classNames.size());
}
