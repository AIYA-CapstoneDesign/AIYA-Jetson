#ifndef AIYA_JETSON_H
#define AIYA_JETSON_H

#include "Thread.h"
#include "videoOutput.h"
#include "videoSource.h"
#include "yolov8.h"

class AIYAApp : public Thread {
public:
  AIYAApp();
  ~AIYAApp();
  void Run();

private:
  videoSource *m_video_source;
  videoOutput *m_webrtc_output;

  yolov8 *m_yolov8;
  yolov8::OverlayFlags m_overlay_flags;
  yolov8::Detection *m_detections;
};

#endif
