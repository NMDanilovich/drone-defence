cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq

CPU_FREQ=2265600
CPU=cpu1 #The number can be changed
sudo sh -c "echo $CPU_FREQ > /sys/devices/system/cpu/$CPU/cpufreq/scaling_min_freq"
sudo sh -c "echo $CPU_FREQ > /sys/devices/system/cpu/$CPU/cpufreq/scaling_max_freq"

GPU_FREQ=1020750000
sh -c "echo $GPU_FREQ > /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq"
sh -c "echo $GPU_FREQ > /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq"

jetson_clock --show