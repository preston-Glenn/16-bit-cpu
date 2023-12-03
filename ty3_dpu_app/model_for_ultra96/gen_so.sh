#! /bin/bash
aarch64-xilinx-linux-gcc --sysroot=$SDKTARGETSYSROOT \
-fPIC -shared ./dpu_tf_yolov3tiny.elf \
-o ./libdpumodeltf_yolov3tiny.so
