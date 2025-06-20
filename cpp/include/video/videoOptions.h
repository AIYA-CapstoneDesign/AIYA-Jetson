/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __VIDEO_OPTIONS_H_
#define __VIDEO_OPTIONS_H_

#include "commandLine.h"
#include "imageFormat.h"

#include "URI.h"

/**
 * The videoOptions struct contains common settings that are used
 * to configure and query videoSource and videoOutput streams.
 * @ingroup video
 */
struct videoOptions {
public:
  /**
   * Constructor using default options.
   */
  videoOptions();

  /**
   * The resource URI of the device, IP stream, or file/directory.
   * @see URI for details about accepted protocols and URI formats.
   */
  URI resource;

  /**
   * Optional path to save the compressed stream to a video file on disk,
   * which is to be used in addition to the primary resource URI above.
   * This option can be set from the command-line using `--input-save`
   * for videoSource streams, or `--output-save` for videoOutput streams.
   */
  URI save;

  /**
   * The width of the stream (in pixels).
   * This option can be set from the command line using `--input-width=N`
   * for videoSource streams, or `--output-width=N` for videoOutput streams.
   */
  uint32_t width;

  /**
   * The height of the stream (in pixels).
   * This option can be set from the command line using `--input-height=N`
   * for videoSource streams, or `--output-height=N` for videoOutput streams.
   */
  uint32_t height;

  /**
   * The framerate of the stream (the default is 30Hz).
   * This option can be set from the command line using `--input-rate=N` or
   * `--output-rate=N` for input and output streams, respectively. The
   * `--framerate=N` option sets it for both.
   */
  float frameRate;

  /**
   * The number of frames that have been captured or output on this interface.
   */
  uint64_t frameCount;

  /**
   * The encoding bitrate for compressed streams (only applies to video codecs
   * like H264/H265). For videoOutput streams, this option can be set from the
   * command line using `--bitrate=N`.
   * @note the default bitrate for encoding output streams is 4Mbps (target
   * VBR).
   */
  uint32_t bitRate;

  /**
   * The number of ring buffers used for threading.
   * This option can be set from the command line using `--num-buffers=N`.
   * @note the default number of ring buffers is 4.
   */
  uint32_t numBuffers;

  /**
   * If true, indicates the buffers are allocated in zeroCopy memory that is
   * mapped to both the CPU and GPU.  Otherwise, the buffers are only accessible
   * from the GPU.
   * @note the default is true (zeroCopy CPU/GPU access enabled).
   */
  bool zeroCopy;

  /**
   * Control the number of loops for videoSource disk-based inputs (for example,
   * the number of times that a video should loop). Other types of streams will
   * ignore it.
   *
   * The following values are are valid:
   *
   *   -1 = loop forever
   *    0 = don't loop
   *   >0 = set number of loops
   *
   * This option can be set from the command line using `--loop=N`.
   * @note by default, looping is disabled (set to `0`).
   */
  int loop;

  /**
   * Number of milliseconds of video to buffer for network RTSP or WebRTC
   * streams. The default setting is 10ms (which is lower than GStreamer's
   * default settings). If you have connection/buffering problems, try
   * increasing the latency setting. It can be set from the command line using
   * `--input-latency=N` or `--output-latency=N`
   */
  int latency;

  /**
   * Device interface types.
   */
  enum DeviceType {
    DEVICE_DEFAULT = 0, /**< Unknown interface type. */
    DEVICE_V4L2,        /**< V4L2 webcam (e.g. `/dev/video0`) */
    DEVICE_CSI,         /**< MIPI CSI camera */
    DEVICE_IP,          /**< IP-based network stream (e.g. RTP/RTSP) */
    DEVICE_FILE,   /**< Disk-based stream from a file or directory of files */
    DEVICE_DISPLAY /**< OpenGL output stream rendered to an attached display */
  };

  /**
   * Indicates the type of device interface used by this stream.
   */
  DeviceType deviceType;

  /**
   * Input/Output stream type.
   */
  enum IoType {
    INPUT = 0, /**< Input stream (e.g. camera, video/image file, ect.) */
    OUTPUT,    /**< Output stream (e.g. display, video/image file, ect.) */
  };

  /**
   * Indicates if this stream is an input or an output.
   */
  IoType ioType;

  /**
   * Settings of the flip method used by MIPI CSI cameras and compressed video
   * inputs.
   */
  enum FlipMethod {
    FLIP_NONE = 0,             /**< Identity (no rotation) */
    FLIP_COUNTERCLOCKWISE,     /**< Rotate counter-clockwise 90 degrees */
    FLIP_ROTATE_180,           /**< Rotate 180 degrees */
    FLIP_CLOCKWISE,            /**< Rotate clockwise 90 degrees */
    FLIP_HORIZONTAL,           /**< Flip horizontally */
    FLIP_UPPER_RIGHT_DIAGONAL, /**< Flip across upper right/lower left diagonal
                                */
    FLIP_VERTICAL,             /**< Flip vertically */
    FLIP_UPPER_LEFT_DIAGONAL,  /**< Flip across upper left/lower right diagonal
                                */
    FLIP_DEFAULT = FLIP_NONE   /**< Default setting (none) */
  };

  /**
   * The flip method controls if and how an input frame is flipped/rotated in
   * pre-processing from a MIPI CSI camera or compressed video input. Other
   * types of streams will ignore this.
   *
   * This option can be set from the command line using `--flip-method=xyz`,
   * where `xyz` is one of the strings below:
   *
   *   - `none`                 (Identity, no rotation)
   *   - `counterclockwise`     (Rotate counter-clockwise 90 degrees)
   *   - `rotate-180`           (Rotate 180 degrees)
   *   - `clockwise`            (Rotate clockwise 90 degrees)
   *   - `horizontal-flip`      (Flip horizontally)
   *   - `vertical-flip`        (Flip vertically)
   *   - `upper-right-diagonal` (Flip across upper right/lower left diagonal)
   *   - `upper-left-diagonal`  (Flip across upper left/lower right diagonal)
   */
  FlipMethod flipMethod;

  /**
   * Video codecs.
   */
  enum Codec {
    CODEC_UNKNOWN = 0, /**< Unknown/unsupported codec */
    CODEC_RAW,         /**< Uncompressed (e.g. RGB) */
    CODEC_H264,        /**< H.264 */
    CODEC_H265,        /**< H.265 */
    CODEC_VP8,         /**< VP8 */
    CODEC_VP9,         /**< VP9 */
    CODEC_MPEG2,       /**< MPEG2 (decode only) */
    CODEC_MPEG4,       /**< MPEG4 (decode only) */
    CODEC_MJPEG        /**< MJPEG */
  };

  /**
   * Indicates the codec used by the stream.
   * This is only really applicable to compressed streams, otherwise it will be
   * `CODEC_RAW`.
   *
   * videoSource input streams will attempt to discover the codec type (i.e.
   * from video file), however RTP streams need this to be explitly set using
   * the `--input-codec=xyz` option (where `xyz` is a string like `h264`,
   * `h265`, `vp8`, `vp9`, `mpeg2`, `mpeg4`, or `mjpeg`).
   *
   * A compressed videoOutput stream will default to H.264 encoding, but can be
   * set using the `--output-codec=xyz` command line option (same values for
   * `xyz` as above).
   */
  Codec codec;

  /**
   * Video codec engines.
   */
  enum CodecType {
    CODEC_CPU = 0, /**< CPU-based implementation using libav (e.g. avdec_h264 /
                      x264enc) */
    CODEC_OMX,     /**< aarch64 & JetPack 4 only - OMX hardware plugins (e.g.
                      omxh264dec/omxh264enc) */
    CODEC_V4L2,    /**< aarch64 & JetPack 5 only - V4L2 hardware plugins (e.g.
                      nvv4l2decoder/nvv4l2h264enc) */
    CODEC_NVENC,   /**< x86 only - NVENC hardware plugin (not currently
                      implemented) */
    CODEC_NVDEC    /**< x86 only - NVDEC hardware plugin (not currently
                      implemented) */
  };

  /**
   * Indicates the underlying hardware/software engine used by the codec.
   * For input streams, this can be set with `--decode=cpu` or `--decode=v4l2`
   * for example. For output streams, this can be set with `--encode=cpu` or
   * `--encode=v4l2` for example. The default setting is to use
   * hardware-acceleration on Jetson (aarch64) and CPU on x86.
   */
  CodecType codecType;

  /**
   * URL of STUN server used for WebRTC.  This can be set using the
   * `--stun-server` command-line argument. STUN servers are used during ICE/NAT
   * and allow a local device to determine its public IP address. If this is
   * left blank and WebRTC is used, then a default STUN server will be assigned.
   */
  std::string stunServer;

  /**
   * Path to a file containing a PEM-encoded SSL/TLS certificate.
   * This is used for enabling HTTPS in the WebRTC server.
   * It can be set from the command-line using the `--ssl-cert` or
   * `--https-cert` options. You can make your own self-signed certificate by
   * running a command like: `openssl req -x509 -newkey rsa:2048 -keyout key.pem
   * -out cert.pem -days 365`
   */
  std::string sslCert;

  /**
   * Path to a file containing a PEM-encoded private key.
   * This is used for enabling HTTPS in the WebRTC server.
   * It can be set from the command-line using the `--ssl-key` or `--https-key`
   * options. You can make your own self-signed certificate by running a command
   * like: `openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem
   * -days 365`
   */
  std::string sslKey;

  /**
   * Log the video settings, with an optional prefix label.
   */
  void Print(const char *prefix = NULL) const;

  /**
   * @internal Parse the video resource URI and options.
   */
  bool Parse(const char *URI, const int argc, char **argv, IoType ioType,
             int ioPositionArg = -1);

  /**
   * @internal Parse the video resource URI and options.
   */
  bool Parse(const char *URI, const commandLine &cmdLine, IoType ioType,
             int ioPositionArg = -1);

  /**
   * @internal Parse the video resource URI and options.
   */
  bool Parse(const int argc, char **argv, IoType ioType,
             int ioPositionArg = -1);

  /**
   * @internal Parse the video resource URI and options.
   */
  bool Parse(const commandLine &cmdLine, IoType ioType, int ioPositionArg = -1);

  /**
   * Convert an IoType enum to a string.
   */
  static const char *IoTypeToStr(IoType type);

  /**
   * Parse an IoType enum from a string.
   */
  static IoType IoTypeFromStr(const char *str);

  /**
   * Convert a DeviceType enum to a string.
   */
  static const char *DeviceTypeToStr(DeviceType type);

  /**
   * Parse a DeviceType enum from a string.
   */
  static DeviceType DeviceTypeFromStr(const char *str);

  /**
   * Convert a FlipMethod enum to a string.
   */
  static const char *FlipMethodToStr(FlipMethod flip);

  /**
   * Parse a FlipMethod enum from a string.
   */
  static FlipMethod FlipMethodFromStr(const char *str);

  /**
   * Convert a Codec enum to a string.
   */
  static const char *CodecToStr(Codec codec);

  /**
   * Parse a Codec enum from a string.
   */
  static Codec CodecFromStr(const char *str);

  /**
   * Convert a CodecType enum to a string.
   */
  static const char *CodecTypeToStr(CodecType codecType);

  /**
   * Parse a Codec enum from a string.
   */
  static CodecType CodecTypeFromStr(const char *str);
};

/**
 * @ingroup video
 */
#define LOG_VIDEO "[video]  "

#endif
