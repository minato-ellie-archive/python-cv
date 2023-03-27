from typing import Literal

import cv2
from pydantic import BaseModel, Field, conint, confloat, validator, root_validator


class VideoCaptureProperties(BaseModel):
    pos_msec: conint(ge=0) = Field(None, alias='CAP_PROP_POS_MSEC')
    pos_frames: conint(ge=0) = Field(None, alias='CAP_PROP_POS_FRAMES')
    pos_avi_ratio: confloat(ge=0) = Field(None, alias='CAP_PROP_POS_AVI_RATIO')
    frame_width: conint(ge=0) = Field(None, alias='CAP_PROP_FRAME_WIDTH')
    frame_height: conint(ge=0) = Field(None, alias='CAP_PROP_FRAME_HEIGHT')
    fps: confloat(ge=0) = Field(None, alias='CAP_PROP_FPS')
    fourcc: conint(ge=0) = Field(None, alias='CAP_PROP_FOURCC')
    frame_count: conint(ge=-1) = Field(None, alias='CAP_PROP_FRAME_COUNT')
    format: conint(ge=0) = Field(None, alias='CAP_PROP_FORMAT')
    mode: conint(ge=0) = Field(None, alias='CAP_PROP_MODE')
    brightness: confloat(ge=0) = Field(None, alias='CAP_PROP_BRIGHTNESS')
    contrast: confloat(ge=0) = Field(None, alias='CAP_PROP_CONTRAST')
    saturation: confloat(ge=0) = Field(None, alias='CAP_PROP_SATURATION')
    hue: confloat(ge=0) = Field(None, alias='CAP_PROP_HUE')
    gain: confloat(ge=0) = Field(None, alias='CAP_PROP_GAIN')
    exposure: confloat(ge=0) = Field(None, alias='CAP_PROP_EXPOSURE')
    convert_rgb: conint(ge=0) = Field(None, alias='CAP_PROP_CONVERT_RGB')
    white_balance_blue_u: confloat(ge=0) = Field(
        None, alias='CAP_PROP_WHITE_BALANCE_BLUE_U')
    rectification: conint(ge=0) = Field(None, alias='CAP_PROP_RECTIFICATION')
    monochrome: conint(ge=0) = Field(None, alias='CAP_PROP_MONOCHROME')
    sharpness: confloat(ge=0) = Field(None, alias='CAP_PROP_SHARPNESS')
    auto_exposure: confloat(ge=0) = Field(None, alias='CAP_PROP_AUTO_EXPOSURE')
    gamma: confloat(ge=0) = Field(None, alias='CAP_PROP_GAMMA')
    temperature: confloat(ge=0) = Field(None, alias='CAP_PROP_TEMPERATURE')
    trigger: confloat(ge=0) = Field(None, alias='CAP_PROP_TRIGGER')
    trigger_delay: confloat(ge=0) = Field(None, alias='CAP_PROP_TRIGGER_DELAY')
    white_balance_red_v: confloat(ge=0) = Field(
        None, alias='CAP_PROP_WHITE_BALANCE_RED_V')
    zoom: confloat(ge=0) = Field(None, alias='CAP_PROP_ZOOM')
    focus: confloat(ge=0) = Field(None, alias='CAP_PROP_FOCUS')
    guid: conint(ge=0) = Field(None, alias='CAP_PROP_GUID')
    iso_speed: confloat(ge=0) = Field(None, alias='CAP_PROP_ISO_SPEED')
    back_light: confloat(ge=0) = Field(None, alias='CAP_PROP_BACKLIGHT')
    pan: confloat(ge=0) = Field(None, alias='CAP_PROP_PAN')
    tilt: confloat(ge=0) = Field(None, alias='CAP_PROP_TILT')
    roll: confloat(ge=0) = Field(None, alias='CAP_PROP_ROLL')
    iris: confloat(ge=0) = Field(None, alias='CAP_PROP_IRIS')
    settings: conint(ge=0) = Field(None, alias='CAP_PROP_SETTINGS')
    buffer_size: conint(ge=0) = Field(None, alias='CAP_PROP_BUFFERSIZE')
    auto_focus: confloat(ge=0) = Field(None, alias='CAP_PROP_AUTOFOCUS')
    sar_num: conint(ge=0) = Field(None, alias='CAP_PROP_SAR_NUM')
    sar_den: conint(ge=0) = Field(None, alias='CAP_PROP_SAR_DEN')
    backend: conint(ge=0) = Field(None, alias='CAP_PROP_BACKEND')
    channel: conint(ge=0) = Field(None, alias='CAP_PROP_CHANNEL')
    auto_wb: confloat(ge=0) = Field(None, alias='CAP_PROP_AUTO_WB')
    wb_temperature: confloat(ge=0) = Field(None,
                                           alias='CAP_PROP_WB_TEMPERATURE')
    codec_pixel_format: conint(ge=0) = Field(
        None, alias='CAP_PROP_CODEC_PIXEL_FORMAT')
    bit_rate: conint(ge=0) = Field(None, alias='CAP_PROP_BITRATE')
    orientation_meta: conint(ge=0) = Field(None,
                                           alias='CAP_PROP_ORIENTATION_META')
    orientation_auto: conint(ge=0) = Field(None,
                                           alias='CAP_PROP_ORIENTATION_AUTO')
    open_timeout: conint(ge=0) = Field(None, alias='CAP_PROP_OPEN_TIMEOUT_MSEC')
    read_timeout: conint(ge=0) = Field(None, alias='CAP_PROP_READ_TIMEOUT_MSEC')

    class Config:
        validate_assignment = True

    @validator('frame_count', pre=True)
    def frame_count_validator(cls, v):
        if v == -1:
            return None
        else:
            assert v >= 0 and isinstance(v, int), 'frame_count must be a positive integer or -1'
            return v


class VideoWriterProperties(BaseModel):
    quality: confloat(ge=0, le=100) = Field(None, alias='VIDEOWRITER_PROP_QUALITY')
    frame_bytes: conint(ge=0) = Field(None, alias='VIDEOWRITER_PROP_FRAMEBYTES')
    n_frames: conint(ge=-1) = Field(None, alias='VIDEOWRITER_PROP_NSTRIPES')

    class Config:
        validate_assignment = True

    @validator('n_frames', pre=True)
    def n_frames_validator(cls, v):
        if v == -1:
            return None
        else:
            assert v >= 0 and isinstance(v, int), 'n_frames must be a positive integer or -1'
            return v


CAPTURE_BACKEND_DICT = {
    'auto': cv2.CAP_ANY,
    'vfw': cv2.CAP_VFW,
    'v4l': cv2.CAP_V4L,
    'v4l2': cv2.CAP_V4L2,
    'firewire': cv2.CAP_FIREWIRE,
    'firewire-1394': cv2.CAP_FIREWIRE,
    'ieee1394': cv2.CAP_IEEE1394,
    'dc1394': cv2.CAP_DC1394,
    'cmu1394': cv2.CAP_CMU1394,
    'quicktime': cv2.CAP_QT,
    'unicap': cv2.CAP_UNICAP,
    'd-show': cv2.CAP_DSHOW,
    'pvapi': cv2.CAP_PVAPI,
    'openni': cv2.CAP_OPENNI,
    'openni-asus': cv2.CAP_OPENNI_ASUS,
    'android': cv2.CAP_ANDROID,
    'xiapi': cv2.CAP_XIAPI,
    'avfoundation': cv2.CAP_AVFOUNDATION,
    'giganetix': cv2.CAP_GIGANETIX,
    'msmf': cv2.CAP_MSMF,
    'winrt': cv2.CAP_WINRT,
    'intel-perc': cv2.CAP_INTELPERC,
    'openni2': cv2.CAP_OPENNI2,
    'openni2-asus': cv2.CAP_OPENNI2_ASUS,
    'gphoto2': cv2.CAP_GPHOTO2,
    'gstreamer': cv2.CAP_GSTREAMER,
    'ffmpeg': cv2.CAP_FFMPEG,
    'images': cv2.CAP_IMAGES,
    'aravis': cv2.CAP_ARAVIS,
    'opencv-mjpeg': cv2.CAP_OPENCV_MJPEG,
    'intel-mfx': cv2.CAP_INTEL_MFX,
    'xine': cv2.CAP_XINE,
}

CaptureBackends = Literal[
    'auto',
    'vfw',
    'v4l',
    'v4l2',
    'firewire',
    'firewire-1394',
    'ieee1394',
    'dc1394',
    'cmu1394',
    'quicktime',
    'unicap',
    'd-show',
    'pvapi',
    'openni',
    'openni-asus',
    'android',
    'xiapi',
    'avfoundation',
    'giganetix',
    'msmf',
    'winrt',
    'intel-perc',
    'openni2',
    'openni2-asus',
    'gphoto2',
    'gstreamer',
    'ffmpeg',
    'images',
    'aravis',
    'opencv-mjpeg',
    'intel-mfx',
    'xine',
]

FourCC = Literal[
    "1",
    "2",
    "2vuy",
    "4",
    "8",
    "8BPS",
    "16",
    "24",
    "24BG",
    "32",
    "33",
    "34",
    "36",
    "40",
    "5551",
    "a2vy",
    "ABGR",
    "ai5p",
    "ai5q",
    "ai52",
    "ai53",
    "ai55",
    "ai56",
    "ai1p",
    "ai1q",
    "ai12",
    "ai13",
    "ai15",
    "ai16",
    "ACTL",
    "ap4h",
    "ap4x",
    "apch",
    "apcn",
    "apco",
    "apcs",
    "avc1",
    "AVdn",
    "AVRn",
    "AVDJ",
    "ADJV",
    "avr",
    "b16g",
    "b32a",
    "b48r",
    "b64a",
    "B565",
    "BGRA",
    "cvid",
    "dmb1",
    "drmi",
    "dv1p",
    "dv1n",
    "dv5n",
    "dv5p",
    "dvc",
    "dvcp",
    "dvh2",
    "dvh3",
    "dvh5",
    "dvh6",
    "dvhp",
    "dvhq",
    "dvp",
    "dvpp",
    "flv",
    "gif",
    "h261",
    "h263",
    "h264",
    "hdv1",
    "hdv2",
    "hdv3",
    "hdv4",
    "hdv5",
    "hdv6",
    "hdv7",
    "hdv8",
    "hdv9",
    "hdva",
    "icod",
    "IV41",
    "IV50",
    "jpeg",
    "Jvt3",
    "L555",
    "L565",
    "mjp2",
    "mjpa",
    "mjpb",
    "mjpg",
    "mpg4",
    "mp1v",
    "mp2v",
    "mp4v",
    "mplo",
    "png",
    "pxlt",
    "r210",
    "r408",
    "RGBA",
    "rle",
    "rpza",
    "s263",
    "smc",
    "theo",
    "v210",
    "v216",
    "v264",
    "v308",
    "v408",
    "v410",
    "VP30",
    "VP31",
    "VP50",
    "VP60",
    "VP70",
    "wmv1",
    "wmv2",
    "wmv3",
    "x264",
    "xd5a",
    "xd59",
    "xdv1",
    "xdv2",
    "xdv3",
    "xdv4",
    "xdv5",
    "xdv6",
    "xdv7",
    "xdv8",
    "xdv9",
    "xdva",
    "xplo",
    "y420",
    "yuvs",
    "zygo",
]
