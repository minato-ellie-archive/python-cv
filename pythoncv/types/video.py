from typing import Literal

import cv2  # type: ignore
from pydantic import (BaseModel, Field, confloat, conint, root_validator, validator)


class VideoCaptureProperties(BaseModel):  # type: ignore
    pos_msec: int = Field(None, alias='CAP_PROP_POS_MSEC', ge=0)
    pos_frames: int = Field(None, alias='CAP_PROP_POS_FRAMES', ge=0)
    pos_avi_ratio: float = Field(None, alias='CAP_PROP_POS_AVI_RATIO', ge=0)
    frame_width: int = Field(None, alias='CAP_PROP_FRAME_WIDTH', ge=0)
    frame_height: int = Field(None, alias='CAP_PROP_FRAME_HEIGHT', ge=0)
    fps: float = Field(None, alias='CAP_PROP_FPS', ge=0)
    fourcc: int = Field(None, alias='CAP_PROP_FOURCC', ge=0)
    frame_count: int = Field(None, alias='CAP_PROP_FRAME_COUNT', ge=-1)
    format: int = Field(None, alias='CAP_PROP_FORMAT', ge=0)
    mode: int = Field(None, alias='CAP_PROP_MODE', ge=0)
    brightness: float = Field(None, alias='CAP_PROP_BRIGHTNESS', ge=0)
    contrast: float = Field(None, alias='CAP_PROP_CONTRAST', ge=0)
    saturation: float = Field(None, alias='CAP_PROP_SATURATION', ge=0)
    hue: float = Field(None, alias='CAP_PROP_HUE', ge=0)
    gain: float = Field(None, alias='CAP_PROP_GAIN', ge=0)
    exposure: float = Field(None, alias='CAP_PROP_EXPOSURE', ge=0)
    convert_rgb: int = Field(None, alias='CAP_PROP_CONVERT_RGB', ge=0)
    white_balance_blue_u: float = Field(None, alias='CAP_PROP_WHITE_BALANCE_BLUE_U', ge=0)
    rectification: int = Field(None, alias='CAP_PROP_RECTIFICATION', ge=0)
    monochrome: int = Field(None, alias='CAP_PROP_MONOCHROME', ge=0)
    sharpness: float = Field(None, alias='CAP_PROP_SHARPNESS', ge=0)
    auto_exposure: float = Field(None, alias='CAP_PROP_AUTO_EXPOSURE', ge=0)
    gamma: float = Field(None, alias='CAP_PROP_GAMMA', ge=0)
    temperature: float = Field(None, alias='CAP_PROP_TEMPERATURE', ge=0)
    trigger: float = Field(None, alias='CAP_PROP_TRIGGER', ge=0)
    trigger_delay: float = Field(None, alias='CAP_PROP_TRIGGER_DELAY', ge=0)
    white_balance_red_v: float = Field(None, alias='CAP_PROP_WHITE_BALANCE_RED_V', ge=0)
    zoom: float = Field(None, alias='CAP_PROP_ZOOM', ge=0)
    focus: float = Field(None, alias='CAP_PROP_FOCUS', ge=0)
    guid: int = Field(None, alias='CAP_PROP_GUID', ge=0)
    iso_speed: float = Field(None, alias='CAP_PROP_ISO_SPEED', ge=0)
    back_light: float = Field(None, alias='CAP_PROP_BACKLIGHT', ge=0)
    pan: float = Field(None, alias='CAP_PROP_PAN', ge=0)
    tilt: float = Field(None, alias='CAP_PROP_TILT', ge=0)
    roll: float = Field(None, alias='CAP_PROP_ROLL', ge=0)
    iris: float = Field(None, alias='CAP_PROP_IRIS', ge=0)
    settings: int = Field(None, alias='CAP_PROP_SETTINGS', ge=0)
    buffer_size: int = Field(None, alias='CAP_PROP_BUFFERSIZE', ge=0)
    auto_focus: float = Field(None, alias='CAP_PROP_AUTOFOCUS', ge=0)
    sar_num: int = Field(None, alias='CAP_PROP_SAR_NUM', ge=0)
    sar_den: int = Field(None, alias='CAP_PROP_SAR_DEN', ge=0)
    backend: int = Field(None, alias='CAP_PROP_BACKEND', ge=0)
    channel: int = Field(None, alias='CAP_PROP_CHANNEL', ge=0)
    auto_wb: float = Field(None, alias='CAP_PROP_AUTO_WB', ge=0)
    wb_temperature: float = Field(None, alias='CAP_PROP_WB_TEMPERATURE', ge=0)
    codec_pixel_format: int = Field(None, alias='CAP_PROP_CODEC_PIXEL_FORMAT', ge=0)
    bit_rate: int = Field(None, alias='CAP_PROP_BITRATE', ge=0)
    orientation_meta: int = Field(None, alias='CAP_PROP_ORIENTATION_META', ge=0)
    orientation_auto: int = Field(None, alias='CAP_PROP_ORIENTATION_AUTO', ge=0)
    open_timeout: int = Field(None, alias='CAP_PROP_OPEN_TIMEOUT_MSEC', ge=0)
    read_timeout: int = Field(None, alias='CAP_PROP_READ_TIMEOUT_MSEC', ge=0)

    class Config:
        validate_assignment = True

    @validator('frame_count', pre=True)
    def frame_count_validator(cls, v):
        if v == -1:
            return None
        else:
            assert v >= 0 and v % 1 == 0.0, \
                'frame_count must be a positive integer or -1'
            return int(v)


class VideoWriterProperties(BaseModel):
    quality: float = Field(None, alias='VIDEOWRITER_PROP_QUALITY', ge=0, le=100)
    frame_bytes: int = Field(None, alias='VIDEOWRITER_PROP_FRAMEBYTES', ge=0)
    n_frames: int = Field(None, alias='VIDEOWRITER_PROP_NSTRIPES', ge=-1)

    class Config:
        validate_assignment = True

    @validator('n_frames', pre=True)
    def n_frames_validator(cls, v):
        if v == -1:
            return None
        else:
            assert v >= 0 and v % 1 == 0.0, \
                'n_frames must be a positive integer or -1'
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

CaptureBackends = Literal['auto', 'vfw', 'v4l', 'v4l2', 'firewire', 'firewire-1394', 'ieee1394', 'dc1394', 'cmu1394',
                          'quicktime', 'unicap', 'd-show', 'pvapi', 'openni', 'openni-asus', 'android', 'xiapi',
                          'avfoundation', 'giganetix', 'msmf', 'winrt', 'intel-perc', 'openni2', 'openni2-asus',
                          'gphoto2', 'gstreamer', 'ffmpeg', 'images', 'aravis', 'opencv-mjpeg', 'intel-mfx', 'xine',]

FourCC = Literal["1", "2", "2vuy", "4", "8", "8BPS", "16", "24", "24BG", "32", "33", "34", "36", "40", "5551", "a2vy",
                 "ABGR", "ai5p", "ai5q", "ai52", "ai53", "ai55", "ai56", "ai1p", "ai1q", "ai12", "ai13", "ai15", "ai16",
                 "ACTL", "ap4h", "ap4x", "apch", "apcn", "apco", "apcs", "avc1", "AVdn", "AVRn", "AVDJ", "ADJV", "avr",
                 "b16g", "b32a", "b48r", "b64a", "B565", "BGRA", "cvid", "dmb1", "drmi", "dv1p", "dv1n", "dv5n", "dv5p",
                 "dvc", "dvcp", "dvh2", "dvh3", "dvh5", "dvh6", "dvhp", "dvhq", "dvp", "dvpp", "flv", "gif", "h261",
                 "h263", "h264", "hdv1", "hdv2", "hdv3", "hdv4", "hdv5", "hdv6", "hdv7", "hdv8", "hdv9", "hdva", "icod",
                 "IV41", "IV50", "jpeg", "Jvt3", "L555", "L565", "mjp2", "mjpa", "mjpb", "mjpg", "mpg4", "mp1v", "mp2v",
                 "mp4v", "mplo", "png", "pxlt", "r210", "r408", "RGBA", "rle", "rpza", "s263", "smc", "theo", "v210",
                 "v216", "v264", "v308", "v408", "v410", "VP30", "VP31", "VP50", "VP60", "VP70", "wmv1", "wmv2", "wmv3",
                 "x264", "xd5a", "xd59", "xdv1", "xdv2", "xdv3", "xdv4", "xdv5", "xdv6", "xdv7", "xdv8", "xdv9", "xdva",
                 "xplo", "y420", "yuvs", "zygo",]
