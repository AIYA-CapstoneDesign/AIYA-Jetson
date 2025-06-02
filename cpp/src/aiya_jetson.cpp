#include "aiya_jetson.h"

AIYAApp::AIYAApp() {
  videoOptions source_options;
  source_options.resource = "/dev/video0";
  source_options.width = 1280;
  source_options.height = 720;
  source_options.frameRate = 30;
  m_video_source = videoSource::Create(source_options);

  videoOptions output_options;
  output_options.resource = "webrtc://@:41567/video";
  output_options.width = 1280;
  output_options.height = 720;
  output_options.frameRate = 30;
  m_webrtc_output = videoOutput::Create(output_options);

  if (!m_video_source) {
    LogError("Failed to create video source");
    exit(1);
  }

  if (!m_webrtc_output) {
    LogError("Failed to create webrtc output");
    exit(1);
  }

  m_webrtc_output->Open();

  m_yolov8 = yolov8::Create("yolov8m.engine", 0.1f);
  m_overlay_flags = yolov8::OverlayFlags::OVERLAY_DEFAULT;
  m_detections = new yolov8::Detection[m_yolov8->GetMaxDetections()];
}

AIYAApp::~AIYAApp() {
  Stop();

  SAFE_DELETE(m_video_source);
  SAFE_DELETE(m_webrtc_output);
}

void AIYAApp::Run() {
  while (mThreadStarted) {
    uchar3 *image = nullptr;
    int status = 0;
    if (!m_video_source->Capture(&image, 1000,
                                 &status)) // 1000ms timeout (default)
    {
      if (status == videoSource::TIMEOUT)
        continue;

      break; // EOS
    }

    m_yolov8->Detect(image, m_video_source->GetWidth(),
                     m_video_source->GetHeight(), m_detections,
                     m_overlay_flags);

    if (m_webrtc_output != NULL) {
      m_webrtc_output->Render(image, m_video_source->GetWidth(),
                              m_video_source->GetHeight());

      if (!m_webrtc_output->IsStreaming()) // check if the user quit
        break;
    }
  }
}