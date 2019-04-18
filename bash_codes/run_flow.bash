docker run -it --rm \
--net=host \
-e DISPLAY=$DISPLAY \
-v /home/syuntoku/flow_rl_project:/headless/rl_project \
-v /home/syuntoku/flow_rl_project/ray_results:/headless/ray_results \
-v /home/syuntoku/.Xauthority:/headless/.Xauthority:rw \
-v /home/syuntoku/flow:/headless/flow \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--entrypoint=/bin/bash \
--shm-size 220G \
syuntoku/flowtorch:latest

# -e NVIDIA_DRIVER_CAPABILITIES=utility,compute \
# -e NVIDIA_VISIBLE_DEVICES=all \
# lucasfischerberkeley/flowdesktop
