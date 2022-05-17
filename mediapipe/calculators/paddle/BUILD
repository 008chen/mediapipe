
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")
cc_library(
    name = "paddle_calculator",
    srcs = ["paddle_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        ":paddle_calculator_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        # "//mediapipe/framework/port:integral_types",
        # "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_imgproc",

    ],
    alwayslink = 1,
)

proto_library(
    name = "paddle_calculator_proto",
    srcs = ["paddle_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "paddle_calculator_cc_proto",
    srcs = ["paddle_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":paddle_calculator_proto"],
)