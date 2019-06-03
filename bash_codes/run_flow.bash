docker run -it --rm \
--net=host \
-e DISPLAY=$DISPLAY \
-v /home/syuntoku14/flow_rl_project:/headless/rl_project \
-v /home/syuntoku14/flow_rl_project/ray_results:/headless/ray_results \
-v /home/syuntoku14/.Xauthority:/headless/.Xauthority:rw \
-v /home/syuntoku14/flow:/headless/flow \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--entrypoint=/bin/bash \
--shm-size 220G \
syuntoku/flowtorch:ray_latest

# -e NVIDIA_DRIVER_CAPABILITIES=utility,compute \
# -e NVIDIA_VISIBLE_DEVICES=all \
# lucasfischerberkeley/flowdesktop
