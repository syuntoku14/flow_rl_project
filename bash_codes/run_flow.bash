docker run --runtime=nvidia \
-it --rm \
--net=host \
-e NVIDIA_DRIVER_CAPABILITIES=utility,compute \
-e NVIDIA_VISIBLE_DEVICES=all \
-e DISPLAY=$DISPLAY \
-v /home/syuntoku/bash_codes:/headless/bash_codes \
-v /home/syuntoku/rl_project:/headless/rl_project \
-v /home/syuntoku/.Xauthority:/headless/.Xauthority:rw \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--entrypoint=/bin/bash \
syuntoku/flowtorch:latest
